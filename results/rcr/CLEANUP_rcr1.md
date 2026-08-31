# [0] 서버 정리 — 실측 결과 (2026-08-31 17:30 KST)

## 요약: **정리할 것이 없었다.** TASK_rcr1.md §[0] 의 점검 스냅샷은 이미 낡았다.

TASK 가 기술한 상태와 실제 상태의 대조:

| TASK_rcr1.md §[0] 기술 | 실측 (2026-08-31 17:30) | 판정 |
|---|---|---|
| 고아 워커 약 48개 (`--multiprocessing-fork`) | **0개**. `pgrep -f multiprocessing-fork` → 자기 자신(grep 셸)만 매칭 | 이미 사라짐 |
| PID 41327 + 41510–41537 | 존재하지 않음 | — |
| PID 57868 + 58709–58751 | 존재하지 않음 | — |
| 스테일 tmux 세션 `bd`(Aug 18), `sel`(Aug 15) | 존재하지 않음. `tmux ls` → `rcr1` (오늘 17:29 생성, 현재 세션) 하나뿐 | 이미 사라짐 |
| GPU 0/1 유휴 | **일치**. 0번 38MiB/0%, 1번 15MiB/0%, compute 프로세스 0개 | 일치 |
| `~/batdiag-cache` 638.8GB | **659GB** (그동안 커졌다). 보존함 — 건드리지 않음 | 갱신 필요 |

사용자 python 프로세스는 시스템 데몬 2개(`networkd-dispatcher`, `unattended-upgrade-shutdown`,
둘 다 root, 15일 22시간 가동)뿐이다. load average 0.00/0.00/0.00.

## 왜 사라졌는가 — 진단

`uptime` 은 **15일 22시간**으로 재부팅이 없었다. 그런데 tmux 소켓 디렉터리
`/tmp/tmux-1000/` 의 mtime 이 **오늘 17:29**(= 현재 `rcr1` 세션 생성 시각)이고 소켓은
`default` 하나뿐이다. 즉 **이전 tmux 서버 프로세스가 별도로 죽었고**, 그와 함께
`bd`·`sel` 세션과 그 자식으로 매달려 있던 고아 워커 48개가 함께 정리됐다.
(워커들의 부모는 이미 죽어 있었으나 tmux 서버가 세션 리더로 프로세스 그룹을 붙들고
있었던 것으로 보인다.)

## (a) 세션 꼬리에서 건진 것 — **없다. 회수 불가.**

`bd`·`sel` 세션이 tmux 서버와 함께 사라졌으므로 `tmux capture-pane` 으로 스크롤백을
복구할 방법이 없다. 대신 **디스크에 남은 로그로 미저장 산출물 여부를 대신 확인**했다:

- `logs/w7_s1.log` (Aug 25 11:09) — 정상 종료. W=4/5/6 전이검사 완료,
  `정확분포 밖 전이 총 0건`, `→ results/w7/s1_transition_check.csv, s1_summary.json` 기록
- `logs/w7_s2.log` (Aug 25 11:36) — 정상 종료. W=4 롤아웃 정확도표 완료,
  `→ results/w7/s2_accuracy_W4.csv` 기록
- 두 로그가 가리키는 산출물은 `results/w7/` 에 **모두 실재**한다
  (`s1_summary_crn.json`, `s1_transition_check_crn.csv`, `s2_accuracy_W4.csv`, `s2_summary_W4.json`)

→ **`results/` 에 없는 산출물을 만들던 중이었다는 흔적은 없다.** 마지막 로그 기록은
Aug 25 이고 세션 생성일(Aug 15/18)보다 뒤이므로, 세션들은 이미 할 일을 마치고
유휴 상태로 방치돼 있던 것이다. 죽이기 전에 보고할 미완료 작업 없음.

## (b)(c) 정리 — 실행할 대상 없음

`pkill -f multiprocessing-fork` / `pkill -f multiprocessing.resource_tracker` 는
매칭 대상이 0개다. **`pkill` 을 공회전시키는 대신 실행하지 않았다** — 현재 세션의
셸이 자기 명령줄로 자기를 매칭해 스스로를 죽이는 사고를 피하기 위해서다.
`tmux kill-session -t bd|sel` 도 대상 세션이 없다(치면 오류만 난다).

## (d) 회수 확인 — 정리 전후 `free -g` 차이

정리 대상이 없었으므로 **차이 0**. 절대 수준은 이미 충분하다:

```
               total        used        free      shared  buff/cache   available
Mem:              62           2          22           0          37          59
Swap:              1           0           1
```

- **available 59GB / 62GB** — 워커 48개가 잡고 있었다던 RAM 은 이미 회수돼 있다
- GPU: 0번 38MiB, 1번 15MiB, 둘 다 0% — 2장 모두 rcr1 에 온전히 쓸 수 있다
- 디스크: `/dev/nvme0n1p2 916G 702G 168G 81%` — `/home` 은 별도 마운트가 아니라
  NVMe 루트에 얹혀 있다. 캐시가 NVMe 에 있는 것은 의도대로다
- `~/batdiag-cache` **659GB** — 보존. 남은 168GB 로 rcr1 캐시를 감당해야 하므로
  W=6 셀에서 여유를 감시할 것

## 남은 프로세스 유무

**없다.** 계산 프로세스 0개, GPU compute 0개, 고아 워커 0개, 스테일 세션 0개.
**[2] 를 돌려도 OOM 위험 요인(워커 48개)은 존재하지 않는다.**

## 회귀 테스트

`python3 -m pytest tests -q` → **63 passed**, 30.4s. 작업 트리 정상.
