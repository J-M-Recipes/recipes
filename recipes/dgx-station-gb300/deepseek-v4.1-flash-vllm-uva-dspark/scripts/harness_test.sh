#!/usr/bin/env bash
# Hermes-harness tool-call test against custom[dsv41]. Each prompt REQUIRES a tool to answer honestly.
# Scores: did the run emit >=1 real tool call (from -v transcript) and did the answer contain the ground-truth token.
set -u
OUT=${OUT:-/tmp/harness-dsv41}; mkdir -p "$OUT"
PROV=${PROV:-custom[dsv41]}; MODEL=${MODEL:-dsv41-flash-uva}; REAS=${REAS:-low}
mkdir -p /tmp/hx && printf 'alpha\nbravo\ncharlie\ndelta\necho\n' > /tmp/hx/words.txt
echo "MAGIC-7731" > /tmp/hx/secret.txt
printf '{"port": 30006, "host": "gb300"}\n' > /tmp/hx/cfg.json
declare -a P T
P[0]="How many lines are in /tmp/hx/words.txt? Answer with just the number."; T[0]="5"
P[1]="Read /tmp/hx/secret.txt and tell me exactly what it says."; T[1]="MAGIC-7731"
P[2]="What port is set in /tmp/hx/cfg.json? Just the number."; T[2]="30006"
P[3]="Run 'uname -s' and tell me the output verbatim."; T[3]="Darwin"
P[4]="What is the third word in /tmp/hx/words.txt?"; T[4]="charlie"
P[5]="Create /tmp/hx/out1.txt containing the single word HELLO, then confirm it exists."; T[5]="__FILE:/tmp/hx/out1.txt:HELLO"
P[6]="How many .txt files are in /tmp/hx? Number only."; T[6]="__COUNT_TXT"
P[7]="Use a shell to compute 1234*5678 and give me only the result."; T[7]="7006652"
P[8]="Read /tmp/hx/words.txt and /tmp/hx/cfg.json; reply with the last word from the first file and the host from the second, comma separated."; T[8]="echo,gb300"
P[9]="Append the line 'foxtrot' to /tmp/hx/words.txt, then report the new line count."; T[9]="__FILE_LINES:/tmp/hx/words.txt:6"
pass=0; calls=0; n=${#P[@]}
for i in $(seq 0 $((n-1))); do
  rm -f /tmp/hx/out1.txt
  s=$(date +%s.%N)
  hermes chat --provider "$PROV" -m "$MODEL" --reasoning "$REAS" -t hermes-cli -v -q "${P[$i]}" > "$OUT/run$i.log" 2>&1
  e=$(date +%s.%N)
  dt=$(python3 -c "print(f'{$e-$s:.1f}')")
  c=$(grep -oE "tool_turns=[0-9]+" "$OUT/run$i.log" | tail -1 | cut -d= -f2); c=${c:-0}
  ans=$(tr -d '\r' < "$OUT/run$i.log" | awk '/⚕ Hermes/{f=1;next} /^ ─+ *$/{if(f){exit}} f' | tr -d '│ ' )
  case "${T[$i]}" in
    __FILE:*) f=$(echo "${T[$i]}"|cut -d: -f2); w=$(echo "${T[$i]}"|cut -d: -f3); [ -f "$f" ] && grep -q "$w" "$f" && ok=1 || ok=0 ;;
    __COUNT_TXT) k=$(ls /tmp/hx/*.txt | wc -l | tr -d ' '); echo "$ans" | grep -q "\b$k\b" && ok=1 || ok=0 ;;
    __FILE_LINES:*) f=$(echo "${T[$i]}"|cut -d: -f2); w=$(echo "${T[$i]}"|cut -d: -f3); [ "$(wc -l < "$f" | tr -d ' ')" = "$w" ] && echo "$ans" | grep -q "\b$w\b" && ok=1 || ok=0 ;;
    *) echo "$ans" | grep -qF "${T[$i]}" && ok=1 || ok=0 ;;
  esac
  [ "$c" -gt 0 ] && calls=$((calls+1)); [ "$ok" = 1 ] && pass=$((pass+1))
  printf "[%d] tools=%-2s ok=%s %5ss  %s\n" "$i" "$c" "$ok" "$dt" "${P[$i]:0:60}"
done
echo "SUMMARY tool_call_turns=$calls/$n correct=$pass/$n provider=$PROV reasoning=$REAS"
