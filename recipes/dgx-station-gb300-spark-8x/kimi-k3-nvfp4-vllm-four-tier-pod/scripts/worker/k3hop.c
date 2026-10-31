/* k3hop — RC SEND/RECV over RoCE, handle-based (many connections per process).
 *
 * Generalised from kimi-k3-pod/g2a-hop/hop_ibv.c (single global QP, always sent the full 16 KiB slot).
 * Changes: (1) every connection is a handle, so one Station process can drive all eight Sparks;
 * (2) sends carry an explicit length (a 7 KiB activation is 2 MTU-4096 packets, not 4);
 * (3) post-send does not block — the send completion is drained by the next poll, and a slot is not
 *     re-posted for send until its previous send completed;
 * (4) the ring buffer can be supplied by the caller (so a worker can cudaHostRegister it and let the
 *     GPU read x / write y in place) — pass NULL to have the library allocate it.
 *
 * Slot 0 is send-only on both sides (never armed for recv), the g2a convention.
 * Build: gcc -O2 -shared -fPIC -o libk3hop.so k3hop.c -libverbs
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* 8 MiB slots: a batched prefill chunk (<= 1,024 tokens x 7,296 B) is one SEND. Decode still sends 7,424 B. */
#define K3_SLOT (8u << 20)
#define K3_NSLOTS 4   /* 4 x 8 MiB: a 1,024-token batch (7.5 MB) is one SEND; decode needs one slot in flight */
#define K3_PORT_IB 1

typedef struct k3conn {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    uint8_t *buf;
    int own_buf;
    int tcp;
    int recv_q[K3_NSLOTS];
    int recv_head, recv_tail, recv_n;
    int send_busy[K3_NSLOTS];
} k3conn;

static __thread char g_err[512];
static void set_err(const char *what) { snprintf(g_err, sizeof(g_err), "%s: %s", what, strerror(errno)); }
const char *k3_last_error(void) { return g_err; }
int k3_nslots(void) { return K3_NSLOTS; }
int k3_slot_bytes(void) { return K3_SLOT; }
size_t k3_ring_bytes(void) { return (size_t)K3_NSLOTS * K3_SLOT; }

void *k3_buf(k3conn *c, int slot) {
    if (!c || slot < 0 || slot >= K3_NSLOTS || !c->buf) return NULL;
    return c->buf + (size_t)slot * K3_SLOT;
}

static int tcp_send_all(int fd, const void *p, size_t n) {
    const uint8_t *b = p; size_t off = 0;
    while (off < n) {
        ssize_t r = send(fd, b + off, n - off, MSG_NOSIGNAL);
        if (r < 0) { if (errno == EINTR) continue; set_err("tcp send"); return -1; }
        off += (size_t)r;
    }
    return 0;
}
static int tcp_recv_all(int fd, void *p, size_t n) {
    uint8_t *b = p; size_t off = 0;
    while (off < n) {
        ssize_t r = recv(fd, b + off, n - off, 0);
        if (r == 0) { snprintf(g_err, sizeof(g_err), "tcp eof"); return -1; }
        if (r < 0) { if (errno == EINTR) continue; set_err("tcp recv"); return -1; }
        off += (size_t)r;
    }
    return 0;
}

struct k3_info { uint32_t lid, qpn, psn, mtu; uint8_t gid[16]; };

static int poll_once(k3conn *c) {
    struct ibv_wc wc[16];
    int n = ibv_poll_cq(c->cq, 16, wc);
    if (n < 0) { snprintf(g_err, sizeof(g_err), "ibv_poll_cq failed"); return -1; }
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(g_err, sizeof(g_err), "wc status %s opcode %d wr_id %llu",
                     ibv_wc_status_str(wc[i].status), wc[i].opcode, (unsigned long long)wc[i].wr_id);
            return -1;
        }
        int slot = (int)(wc[i].wr_id & 0xffffffffu);
        if (wc[i].opcode == IBV_WC_RECV) {
            if (c->recv_n >= K3_NSLOTS) { snprintf(g_err, sizeof(g_err), "recv queue overflow"); return -1; }
            c->recv_q[c->recv_tail] = slot;
            c->recv_tail = (c->recv_tail + 1) % K3_NSLOTS;
            c->recv_n++;
        } else if (wc[i].opcode == IBV_WC_SEND) {
            if (slot >= 0 && slot < K3_NSLOTS) c->send_busy[slot] = 0;
        } else {
            snprintf(g_err, sizeof(g_err), "unexpected wc opcode %d", wc[i].opcode);
            return -1;
        }
    }
    return n;
}

int k3_post_recv(k3conn *c, int slot) {
    if (slot <= 0 || slot >= K3_NSLOTS) { snprintf(g_err, sizeof(g_err), "bad recv slot %d", slot); return -1; }
    struct ibv_sge sge = {.addr = (uintptr_t)k3_buf(c, slot), .length = K3_SLOT, .lkey = c->mr->lkey};
    struct ibv_recv_wr wr = {.wr_id = (uint64_t)slot, .sg_list = &sge, .num_sge = 1};
    struct ibv_recv_wr *bad = NULL;
    if (ibv_post_recv(c->qp, &wr, &bad)) { set_err("ibv_post_recv"); return -1; }
    return 0;
}

/* Post a send of `len` bytes from `slot`. Waits only if that slot's previous send is still in flight. */
int k3_send(k3conn *c, int slot, int len) {
    if (slot < 0 || slot >= K3_NSLOTS || len <= 0 || len > K3_SLOT) {
        snprintf(g_err, sizeof(g_err), "bad send slot %d len %d", slot, len); return -1;
    }
    struct timespec t0, now; clock_gettime(CLOCK_MONOTONIC, &t0);
    while (c->send_busy[slot]) {
        if (poll_once(c) < 0) return -1;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if ((now.tv_sec - t0.tv_sec) * 1e3 + (now.tv_nsec - t0.tv_nsec) / 1e6 > 2000.0) {
            snprintf(g_err, sizeof(g_err), "send slot %d still busy after 2 s", slot); return -1;
        }
    }
    struct ibv_sge sge = {.addr = (uintptr_t)k3_buf(c, slot), .length = (uint32_t)len, .lkey = c->mr->lkey};
    struct ibv_send_wr wr = {.wr_id = (uint64_t)slot, .sg_list = &sge, .num_sge = 1,
                             .opcode = IBV_WR_SEND, .send_flags = IBV_SEND_SIGNALED};
    struct ibv_send_wr *bad = NULL;
    c->send_busy[slot] = 1;
    if (ibv_post_send(c->qp, &wr, &bad)) { c->send_busy[slot] = 0; set_err("ibv_post_send"); return -1; }
    return 0;
}

/* Next received slot, or -1 on timeout/error (check k3_last_error: empty string means timeout). */
int k3_poll_recv(k3conn *c, int timeout_us) {
    struct timespec t0, now; int first = 1;
    g_err[0] = 0;
    for (;;) {
        if (c->recv_n > 0) {
            int slot = c->recv_q[c->recv_head];
            c->recv_head = (c->recv_head + 1) % K3_NSLOTS;
            c->recv_n--;
            return slot;
        }
        if (poll_once(c) < 0) return -1;
        if (c->recv_n > 0) continue;
        if (first) { clock_gettime(CLOCK_MONOTONIC, &t0); first = 0; continue; }
        clock_gettime(CLOCK_MONOTONIC, &now);
        if ((now.tv_sec - t0.tv_sec) * 1e6 + (now.tv_nsec - t0.tv_nsec) / 1e3 >= timeout_us) return -1;
    }
}

static k3conn *setup(const char *ibdev, int gid_index, void *ring, struct k3_info *local) {
    k3conn *c = calloc(1, sizeof(*c));
    if (!c) { set_err("calloc"); return NULL; }
    c->tcp = -1;
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) { set_err("ibv_get_device_list"); free(c); return NULL; }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < n; i++) if (strcmp(ibv_get_device_name(list[i]), ibdev) == 0) dev = list[i];
    if (!dev) { snprintf(g_err, sizeof(g_err), "device %s not found", ibdev); ibv_free_device_list(list); free(c); return NULL; }
    c->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (!c->ctx) { set_err("ibv_open_device"); free(c); return NULL; }
    struct ibv_port_attr pattr;
    if (ibv_query_port(c->ctx, K3_PORT_IB, &pattr) || pattr.state != IBV_PORT_ACTIVE) {
        snprintf(g_err, sizeof(g_err), "port not active"); return NULL;
    }
    union ibv_gid gid;
    if (ibv_query_gid(c->ctx, K3_PORT_IB, gid_index, &gid)) { set_err("ibv_query_gid"); return NULL; }
    c->pd = ibv_alloc_pd(c->ctx);
    if (!c->pd) { set_err("ibv_alloc_pd"); return NULL; }
    size_t bytes = k3_ring_bytes();
    if (ring) { c->buf = ring; c->own_buf = 0; }
    else {
        c->buf = aligned_alloc(4096, bytes);
        if (!c->buf) { set_err("aligned_alloc"); return NULL; }
        memset(c->buf, 0, bytes); c->own_buf = 1;
    }
    c->mr = ibv_reg_mr(c->pd, c->buf, bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!c->mr) { set_err("ibv_reg_mr"); return NULL; }
    c->cq = ibv_create_cq(c->ctx, K3_NSLOTS * 8, NULL, NULL, 0);
    if (!c->cq) { set_err("ibv_create_cq"); return NULL; }
    struct ibv_qp_init_attr qia = {
        .send_cq = c->cq, .recv_cq = c->cq,
        .cap = {.max_send_wr = K3_NSLOTS * 2, .max_recv_wr = K3_NSLOTS * 2, .max_send_sge = 1, .max_recv_sge = 1},
        .qp_type = IBV_QPT_RC, .sq_sig_all = 0,
    };
    c->qp = ibv_create_qp(c->pd, &qia);
    if (!c->qp) { set_err("ibv_create_qp"); return NULL; }
    struct ibv_qp_attr attr; memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT; attr.port_num = K3_PORT_IB;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(c->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        set_err("modify INIT"); return NULL;
    }
    for (int i = 1; i < K3_NSLOTS; i++) if (k3_post_recv(c, i) != 0) return NULL;
    local->lid = pattr.lid; local->qpn = c->qp->qp_num;
    local->psn = (uint32_t)(lrand48() & 0xffffff); local->mtu = pattr.active_mtu;
    memcpy(local->gid, gid.raw, 16);
    return c;
}

static int connect_qp(k3conn *c, const struct k3_info *local, const struct k3_info *remote, int gid_index) {
    struct ibv_qp_attr attr; memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = local->mtu < remote->mtu ? local->mtu : remote->mtu;
    attr.dest_qp_num = remote->qpn; attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 1; attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1; attr.ah_attr.dlid = remote->lid; attr.ah_attr.port_num = K3_PORT_IB;
    memcpy(attr.ah_attr.grh.dgid.raw, remote->gid, 16);
    attr.ah_attr.grh.sgid_index = gid_index; attr.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(c->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        set_err("modify RTR"); return -1;
    }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS; attr.timeout = 14; attr.retry_cnt = 7; attr.rnr_retry = 7;
    attr.sq_psn = local->psn; attr.max_rd_atomic = 1;
    if (ibv_modify_qp(c->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                      IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        set_err("modify RTS"); return -1;
    }
    return 0;
}

static int exchange(int fd, struct k3_info *local, struct k3_info *remote) {
    if (tcp_send_all(fd, local, sizeof(*local)) || tcp_recv_all(fd, remote, sizeof(*remote))) return -1;
    char rdy[4] = {'R', 'D', 'Y', '\n'}, got[4];
    if (tcp_send_all(fd, rdy, 4) || tcp_recv_all(fd, got, 4)) return -1;
    if (memcmp(got, rdy, 4)) { snprintf(g_err, sizeof(g_err), "bad ready token"); return -1; }
    return 0;
}

k3conn *k3_listen(const char *bind_ip, int port, const char *ibdev, int gid_index, void *ring, int accept_timeout_s) {
    srand48(time(NULL) ^ getpid());
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { set_err("socket"); return NULL; }
    int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(port)};
    if (inet_pton(AF_INET, bind_ip, &addr.sin_addr) != 1) { snprintf(g_err, sizeof(g_err), "bad bind ip"); close(srv); return NULL; }
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) || listen(srv, 1)) { set_err("bind/listen"); close(srv); return NULL; }
    struct k3_info local, remote; memset(&local, 0, sizeof(local));
    k3conn *c = setup(ibdev, gid_index, ring, &local);
    if (!c) { close(srv); return NULL; }
    fprintf(stderr, "K3HOP_LISTEN %s:%d qpn=%u mtu=%u\n", bind_ip, port, local.qpn, local.mtu); fflush(stderr);
    struct timeval tv = {.tv_sec = accept_timeout_s, .tv_usec = 0};
    setsockopt(srv, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    int fd = accept(srv, NULL, NULL);
    close(srv);
    if (fd < 0) { set_err("accept"); return NULL; }
    int nd = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
    c->tcp = fd;
    if (exchange(fd, &local, &remote) || connect_qp(c, &local, &remote, gid_index)) return NULL;
    fprintf(stderr, "K3HOP_RTS peer_qpn=%u\n", remote.qpn); fflush(stderr);
    return c;
}

k3conn *k3_connect(const char *peer_ip, int port, const char *ibdev, int gid_index, void *ring) {
    srand48(time(NULL) ^ getpid() ^ port);
    struct k3_info local, remote; memset(&local, 0, sizeof(local));
    k3conn *c = setup(ibdev, gid_index, ring, &local);
    if (!c) return NULL;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { set_err("socket"); return NULL; }
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(port)};
    if (inet_pton(AF_INET, peer_ip, &addr.sin_addr) != 1) { snprintf(g_err, sizeof(g_err), "bad peer ip"); close(fd); return NULL; }
    int nd = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
    struct timeval tv = {.tv_sec = 30, .tv_usec = 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)); setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr))) { set_err("connect"); close(fd); return NULL; }
    c->tcp = fd;
    if (exchange(fd, &local, &remote) || connect_qp(c, &local, &remote, gid_index)) return NULL;
    return c;
}

int k3_close(k3conn *c) {
    if (!c) return 0;
    if (c->qp) ibv_destroy_qp(c->qp);
    if (c->cq) ibv_destroy_cq(c->cq);
    if (c->mr) ibv_dereg_mr(c->mr);
    if (c->pd) ibv_dealloc_pd(c->pd);
    if (c->ctx) ibv_close_device(c->ctx);
    if (c->own_buf) free(c->buf);
    if (c->tcp >= 0) close(c->tcp);
    free(c);
    return 0;
}

/* ---------------------------------------------------------------------------------------------------------
 * Pod fast path (Station side, one token): pack + post a layer's dispatch to every worker that holds >= 1 of
 * the 16 routed ids, and collect the replies straight into a [nconn][hidden] bf16 row buffer.
 * Wire format: k3proto.py v1 (HDR = 256; REQ "<I H B B I 16i 16f"; REP "<I I H H I Q Q Q Q Q I").
 * ------------------------------------------------------------------------------------------------------- */
#define K3_HDR 256

/* Returns a bitmask of target connections (bit wi). k_expect[wi] = ids this connection must use.
 * *n_station = ids owned by nobody remote (owner < 0, or id out of range). Negative return on error. */
long long k3pod_send(k3conn **conns, int nconn, const int8_t *owner, const int32_t *ids, const float *w,
                     const uint16_t *x, int hidden, int layer, uint32_t rid_base, int8_t *k_expect, int *n_station) {
    long long mask = 0;
    int ns = 0;
    for (int i = 0; i < nconn; i++) k_expect[i] = 0;
    for (int j = 0; j < 16; j++) {
        int g = ids[j];
        int o = (g >= 0 && g < 896) ? owner[g] : -2;
        if (o >= 0 && o < nconn) { k_expect[o]++; mask |= (1LL << o); }
        else ns++;
    }
    *n_station = ns;
    size_t xbytes = (size_t)hidden * 2;
    if (K3_HDR + xbytes > K3_SLOT) { snprintf(g_err, sizeof(g_err), "hidden too large"); return -1; }
    for (int wi = 0; wi < nconn; wi++) {
        if (!(mask & (1LL << wi))) continue;
        uint8_t *b = k3_buf(conns[wi], 0);
        /* the previous send from slot 0 must be complete before we overwrite it */
        struct timespec t0, now; clock_gettime(CLOCK_MONOTONIC, &t0);
        while (conns[wi]->send_busy[0]) {
            if (poll_once(conns[wi]) < 0) return -2;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if ((now.tv_sec - t0.tv_sec) * 1e3 + (now.tv_nsec - t0.tv_nsec) / 1e6 > 2000.0) {
                snprintf(g_err, sizeof(g_err), "slot 0 busy on conn %d", wi); return -3;
            }
        }
        uint32_t rid = rid_base + (uint32_t)wi;
        uint16_t L = (uint16_t)layer;
        uint8_t ntok = 1, flags = 0; uint32_t tag = 0;
        memcpy(b + 0, &rid, 4); memcpy(b + 4, &L, 2); b[6] = ntok; b[7] = flags; memcpy(b + 8, &tag, 4);
        memcpy(b + 12, ids, 64); memcpy(b + 76, w, 64);
        memcpy(b + K3_HDR, x, xbytes);
        if (k3_send(conns[wi], 0, (int)(K3_HDR + xbytes)) != 0) return -4;
    }
    return mask;
}

/* Collect replies for `mask`; row wi of out gets worker wi's bf16 y (zeroed if not a target).
 * Returns total k_used, or negative on error (status, k mismatch, timeout). */
int k3pod_collect(k3conn **conns, int nconn, long long mask, const int8_t *k_expect, uint16_t *out,
                  int hidden, int timeout_us, uint32_t rid_base) {
    int total = 0;
    size_t xbytes = (size_t)hidden * 2;
    for (int wi = 0; wi < nconn; wi++) {
        if (!(mask & (1LL << wi))) { memset(out + (size_t)wi * hidden, 0, xbytes); continue; }
        int s = k3_poll_recv(conns[wi], timeout_us);
        if (s < 0) { if (!g_err[0]) snprintf(g_err, sizeof(g_err), "timeout on conn %d", wi); return -1; }
        uint8_t *rv = k3_buf(conns[wi], s);
        uint32_t rq, st; uint16_t k;
        memcpy(&rq, rv, 4); memcpy(&st, rv + 4, 4); memcpy(&k, rv + 8, 2);
        if (rq != rid_base + (uint32_t)wi) { snprintf(g_err, sizeof(g_err), "conn %d req_id %u != %u", wi, rq, rid_base + wi); return -2; }
        if (st != 0) { snprintf(g_err, sizeof(g_err), "conn %d status %u", wi, st); return -3; }
        if ((int)k != k_expect[wi]) { snprintf(g_err, sizeof(g_err), "conn %d k_used %u != %d", wi, k, k_expect[wi]); return -4; }
        memcpy(out + (size_t)wi * hidden, rv + K3_HDR, xbytes);
        if (k3_post_recv(conns[wi], s) != 0) return -5;
        total += k;
    }
    return total;
}

/* ---------------------------------------------------------------------------------------------------------
 * Graph-resident expert step (idea: Javier, @javierbits). The GPU, inside a CUDA graph, writes x/ids/w for layer L
 * into pinned host memory and then sets ready[L]=1 (cuStreamWriteValue32); this proxy thread sees it, clears it,
 * sends + collects, writes the reply rows into pinned host memory and sets done[L]=1; the graph waits on done[L]==1
 * (cuStreamWaitValue32) and resets it to 0. Constant flag values -> a fixed graph replays forever.
 * Any failure aborts the process: a wrong MoE output must never pass silently.
 * nconn == 0: loopback test mode (rows[L][0] = x[L], rest zero).
 * ------------------------------------------------------------------------------------------------------- */
#include <pthread.h>
#include <sched.h>

#define K3P_FLAG_STRIDE 32   /* int32 words per layer: ready at [0], done at [16] -> separate 64 B lines */
#define K3P_MAXM 8           /* multi-token decode: up to 8 tokens per step through the proxy (ready = M) */

typedef struct {
    k3conn **conns; int nconn; int nlayers; int hidden;
    const int8_t *owner;           /* [nlayers][896] */
    volatile int32_t *flags;       /* [nlayers][32] */
    const uint16_t *x;             /* [nlayers][hidden] bf16 */
    const int32_t *ids;            /* [nlayers][16] */
    const float *w;                /* [nlayers][16] */
    uint16_t *rows;                /* [nlayers][max(nconn,1)][hidden] bf16 */
    /* K3-5 multi-token decode (ready flag value M = 2..K3P_MAXM): */
    const uint16_t *xb;            /* [nlayers][K3P_MAXM][hidden] bf16 */
    const int32_t *idsb;           /* [nlayers][K3P_MAXM][16] */
    const float *wb;               /* [nlayers][K3P_MAXM][16] */
    float *outb;                   /* [nlayers][K3P_MAXM][hidden] f32: sum over workers */
    int timeout_us;
    volatile int stop;
    volatile long long served;
    double t_work;                 /* seconds spent between seeing ready and setting done */
} k3proxy;

static k3proxy g_px;

/* Fail closed and FAST: SIGKILL our own process. abort() was measured to leave the engine core stuck writing a core
 * dump of ~480 GB of mapped memory through apport while the API server kept answering /health 200 and hung every
 * request (fault injection, 2026-10-30 23:10). A SIGKILL'd engine core is detected by vLLM at once. */
#include <signal.h>
#include <unistd.h>
static void k3proxy_die(void) { kill(getpid(), SIGKILL); _exit(137); }
static pthread_t g_px_thr;
static int g_px_running = 0;

static double k3p_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* Multi-token step for layer L with M tokens: one FLAG_BATCH message per worker carrying the tokens it serves (same
 * wire format as k3proto.py batch_*), collect, check k_used per token and Station + workers == 16 per token, and sum
 * the replies into outb[L][t] (f32). Returns 0, or <0 with g_err set. */
static int k3proxy_batch(k3proxy *p, int L, int M, uint32_t rid) {
    int nconn = p->nconn, H = p->hidden;
    if (nconn > 64 || M < 1 || M > K3P_MAXM) { snprintf(g_err, sizeof(g_err), "batch bad nconn %d / M %d", nconn, M); return -1; }
    const int8_t *own = p->owner + (size_t)L * 896;
    const int32_t *ids = p->idsb + (size_t)L * K3P_MAXM * 16;
    const float *w = p->wb + (size_t)L * K3P_MAXM * 16;
    const uint16_t *x = p->xb + (size_t)L * K3P_MAXM * H;
    float *out = p->outb + (size_t)L * K3P_MAXM * H;
    int8_t cnt[64][K3P_MAXM]; int nst[K3P_MAXM], got[K3P_MAXM], rows[64][K3P_MAXM], m[64];
    memset(cnt, 0, sizeof(cnt));
    for (int t = 0; t < M; t++) {
        nst[t] = 0; got[t] = 0;
        for (int j = 0; j < 16; j++) {
            int g = ids[t * 16 + j];
            int o = (g >= 0 && g < 896) ? own[g] : -2;
            if (o >= 0 && o < nconn) cnt[o][t]++; else nst[t]++;
        }
    }
    size_t xbytes = (size_t)H * 2;
    for (int wi = 0; wi < nconn; wi++) {
        m[wi] = 0;
        for (int t = 0; t < M; t++) if (cnt[wi][t]) rows[wi][m[wi]++] = t;
        if (!m[wi]) continue;
        int mm = m[wi];
        size_t len = 256 + (size_t)128 * mm + xbytes * mm;
        if (len > K3_SLOT) { snprintf(g_err, sizeof(g_err), "batch too large"); return -2; }
        k3conn *c = p->conns[wi];
        struct timespec t0, now; clock_gettime(CLOCK_MONOTONIC, &t0);
        while (c->send_busy[0]) {
            if (poll_once(c) < 0) return -3;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if ((now.tv_sec - t0.tv_sec) * 1e3 + (now.tv_nsec - t0.tv_nsec) / 1e6 > 2000.0) {
                snprintf(g_err, sizeof(g_err), "slot 0 busy on conn %d", wi); return -4;
            }
        }
        uint8_t *b = k3_buf(c, 0);
        uint32_t r = rid + (uint32_t)wi; uint16_t LL = (uint16_t)L, M16 = (uint16_t)mm; uint32_t tag = 0;
        memcpy(b + 0, &r, 4); memcpy(b + 4, &LL, 2); b[6] = 0; b[7] = 2 /* FLAG_BATCH */; memcpy(b + 8, &tag, 4);
        memset(b + 12, 0xff, 64); memset(b + 76, 0, 64);           /* unused per-token header fields */
        memcpy(b + 140, &M16, 2);
        for (int k = 0; k < mm; k++) {
            int t = rows[wi][k];
            memcpy(b + 256 + 64 * k, ids + t * 16, 64);
            memcpy(b + 256 + 64 * mm + 64 * k, w + t * 16, 64);
            memcpy(b + 256 + 128 * mm + xbytes * k, x + (size_t)t * H, xbytes);
        }
        if (k3_send(c, 0, (int)len) != 0) return -5;
    }
    memset(out, 0, sizeof(float) * (size_t)M * H);
    for (int wi = 0; wi < nconn; wi++) {
        if (!m[wi]) continue;
        int mm = m[wi];
        k3conn *c = p->conns[wi];
        int s = k3_poll_recv(c, p->timeout_us);
        if (s < 0) { if (!g_err[0]) snprintf(g_err, sizeof(g_err), "batch timeout on conn %d", wi); return -6; }
        uint8_t *rv = k3_buf(c, s);
        uint32_t rq, st; memcpy(&rq, rv, 4); memcpy(&st, rv + 4, 4);
        if (rq != rid + (uint32_t)wi) { snprintf(g_err, sizeof(g_err), "batch conn %d req_id %u != %u", wi, rq, rid + wi); return -7; }
        if (st != 0) { snprintf(g_err, sizeof(g_err), "batch conn %d status %u", wi, st); return -8; }
        const uint16_t *ku = (const uint16_t *)(rv + 256);
        size_t yoff = 256 + ((size_t)(2 * mm + 127) / 128) * 128;
        for (int k = 0; k < mm; k++) {
            int t = rows[wi][k];
            if ((int)ku[k] != cnt[wi][t]) {
                snprintf(g_err, sizeof(g_err), "batch conn %d token %d k_used %u != %d", wi, t, ku[k], cnt[wi][t]); return -9;
            }
            got[t] += ku[k];
            const uint16_t *y = (const uint16_t *)(rv + yoff + xbytes * k);
            float *o = out + (size_t)t * H;
            for (int h = 0; h < H; h++) { union { uint32_t u; float f; } v; v.u = (uint32_t)y[h] << 16; o[h] += v.f; }
        }
        if (k3_post_recv(c, s) != 0) return -10;
    }
    for (int t = 0; t < M; t++)
        if (got[t] + nst[t] != 16) { snprintf(g_err, sizeof(g_err), "batch partition token %d: %d + %d != 16", t, got[t], nst[t]); return -11; }
    return 0;
}

static void *k3proxy_main(void *arg) {
    k3proxy *p = (k3proxy *)arg;
    int8_t kexp[64];
    uint32_t rid = 0x40000000u;
    int L = 0;
    int nrows = p->nconn > 0 ? p->nconn : 1;
    size_t hb = (size_t)p->hidden * 2;
    while (!p->stop) {
        /* scan from the last served layer: the next one is almost always L+1 */
        int found = -1;
        for (int i = 0; i < p->nlayers; i++) {
            int l = (L + i) % p->nlayers;
            if (__atomic_load_n(&p->flags[l * K3P_FLAG_STRIDE], __ATOMIC_ACQUIRE) > 0) { found = l; break; }
        }
        if (found < 0) { __asm__ volatile("yield" ::: "memory"); continue; }
        L = found;
        double t0 = k3p_now();
        int Mtok = __atomic_load_n(&p->flags[L * K3P_FLAG_STRIDE], __ATOMIC_ACQUIRE);
        __atomic_store_n(&p->flags[L * K3P_FLAG_STRIDE], 0, __ATOMIC_RELAXED);
        uint16_t *out = p->rows + (size_t)L * nrows * p->hidden;
        if (Mtok > 1) {
            if (!p->outb || p->nconn == 0) { fprintf(stderr, "k3proxy FATAL: M=%d but no batch regions\n", Mtok); fflush(stderr); k3proxy_die(); }
            rid += (uint32_t)p->nconn;
            if (k3proxy_batch(p, L, Mtok, rid) != 0) {
                fprintf(stderr, "k3proxy FATAL: batch layer %d M=%d: %s\n", L, Mtok, g_err); fflush(stderr); k3proxy_die();
            }
        } else if (p->nconn == 0) {
            memcpy(out, p->x + (size_t)L * p->hidden, hb);
        } else {
            int nst = 0;
            rid += (uint32_t)p->nconn;
            long long m = k3pod_send(p->conns, p->nconn, p->owner + (size_t)L * 896, p->ids + L * 16, p->w + L * 16,
                                     p->x + (size_t)L * p->hidden, p->hidden, L, rid, kexp, &nst);
            if (m < 0) { fprintf(stderr, "k3proxy FATAL: send layer %d: %s\n", L, g_err); fflush(stderr); k3proxy_die(); }
            int got = k3pod_collect(p->conns, p->nconn, m, kexp, out, p->hidden, p->timeout_us, rid);
            if (got < 0) { fprintf(stderr, "k3proxy FATAL: collect layer %d: %s\n", L, g_err); fflush(stderr); k3proxy_die(); }
            if (got + nst != 16) {
                fprintf(stderr, "k3proxy FATAL: partition layer %d: %d + %d != 16\n", L, got, nst); fflush(stderr); k3proxy_die();
            }
        }
        p->t_work += k3p_now() - t0;
        p->served++;
        __atomic_store_n(&p->flags[L * K3P_FLAG_STRIDE + 16], 1, __ATOMIC_RELEASE);
        L = (L + 1) % p->nlayers;
    }
    return NULL;
}

int k3proxy_start(k3conn **conns, int nconn, int nlayers, int hidden, const int8_t *owner, int32_t *flags,
                  const uint16_t *x, const int32_t *ids, const float *w, uint16_t *rows, int timeout_us, int cpu) {
    if (g_px_running) { snprintf(g_err, sizeof(g_err), "proxy already running"); return -1; }
    memset(&g_px, 0, sizeof(g_px));
    g_px.conns = conns; g_px.nconn = nconn; g_px.nlayers = nlayers; g_px.hidden = hidden; g_px.owner = owner;
    g_px.flags = flags; g_px.x = x; g_px.ids = ids; g_px.w = w; g_px.rows = rows; g_px.timeout_us = timeout_us;
    if (pthread_create(&g_px_thr, NULL, k3proxy_main, &g_px) != 0) { snprintf(g_err, sizeof(g_err), "pthread_create"); return -2; }
    if (cpu >= 0) {
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
        pthread_setaffinity_np(g_px_thr, sizeof(s), &s);
    }
    g_px_running = 1;
    return 0;
}

/* K3-5: register the multi-token regions (call after k3proxy_start, before any ready flag > 1 is raised). */
int k3proxy_set_batch(const uint16_t *xb, const int32_t *idsb, const float *wb, float *outb) {
    g_px.xb = xb; g_px.idsb = idsb; g_px.wb = wb;
    __atomic_store_n(&g_px.outb, outb, __ATOMIC_RELEASE);
    return K3P_MAXM;
}

long long k3proxy_served(void) { return g_px.served; }
double k3proxy_work_s(void) { double t = g_px.t_work; g_px.t_work = 0; return t; }

int k3proxy_stop(void) {
    if (!g_px_running) return 0;
    g_px.stop = 1; pthread_join(g_px_thr, NULL); g_px_running = 0; return 0;
}
