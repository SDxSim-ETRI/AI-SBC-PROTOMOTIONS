# AI-SBC 인수인계 — 처음 오는 에이전트가 먼저 읽을 문서 (2026-10-13 기준)

이 레포를 처음 보는 사람·에이전트가 **지금 어디까지 왔고, 무엇이 확정됐고, 무엇이 열려 있는지**를
한 번에 파악하도록 쓴 요약이다. 세부 수치와 판단 근거는 아래 문서로 내려간다.
기존 문서·커밋·run 이름에 남아 있는 옛 이름은 **v13 → L1, v33 → W1, v32 → W0**으로 읽는다.

읽는 순서:

| 순서 | 문서 | 내용 |
|---|---|---|
| 1 | 이 문서 | 전체 상황 요약 |
| 2 | [`AI_SBC_VERSIONS.md`](AI_SBC_VERSIONS.md) (브라우저용 [`versions.html`](versions.html)) | 버전 체계·용어·지금 쓸 것 |
| 3 | [`NEXT_SESSION.md`](NEXT_SESSION.md) | 실험 일지(시간순). 각 실험의 수치·판단·중단 이유 |
| 4 | [`AI_SBC_FRAMEWORK.md`](AI_SBC_FRAMEWORK.md) | 설계 원리, 파일 지도(§4), 함정 모음(§8), 코드 읽기 체크리스트(§9) |
| 5 | [`../examples/experiments/assist_skeleton/README.md`](../examples/experiments/assist_skeleton/README.md) | 통합 시스템(스켈레톤 A/B) 사용법 |
| 6 | [`../CLAUDE.md`](../CLAUDE.md) | 명령·코드 규약·gotcha |
| 7 | `../checkpoints/*/INFO.md` | 보관본별 학습 설정·결과·사용 명령 |

---

## 2026-10-13 — GPU 검증 완료: 새 참조와 W1의 호환성부터 해결 (최신 인계)

**Full access에서 GPU 평가를 실제 완료했다. 접근 제한·승인 대기가 아니다.**
`nvidia-smi`, sbc CUDA 텐서 계산, Newton 평가 모두 정상이다. 이전의 접근 오류 기록을
현재 제한으로 단정하고 중단한 설명을 정정한다. 현재 권한·실제 실행 결과로 판단할 것.
상세 NEXT_SESSION **§2-AA**, [갱신된 보고서](walking_retarget_report.html).

- W1 무보조, 물리 400 Hz/정책 20 Hz, 시작 0/0.75초: 기존 걷기는 두 조건 모두 8초 유지,
  평균 몸체 오차 62/64 mm. 새 느림·보통 8클립×2시작=16조건은 15조건에서 1.35–3.10초에 낙상.
  나머지 A445 보통/0.75초도 골반 약 0.13 m, 낮은 상태 0.45초에서 클립이 끝나 성공이 아니다.
- 빠른 후보 3개×2시작은 5/6조건 낙상. A417/0.75초만 3.15초 구간 종료, 몸체 오차 약 200 mm.
  짧은 구간 종료를 일반적인 추종 성공이나 빠른 걷기/달리기 검증으로 해석하지 않는다.
- **기존 걷기의 참조 표현만 바꿔도 낙상:** root 상태·dps/dvs·접촉·시간은 그대로 두고,
  비root 몸체 회전/각속도만 FK로 교체→2.30/2.65초, 위치/선속도만→1.75/2.40초,
  둘 다→1.35/1.45초 낙상. 원래 참조→8/8초 유지. 6비교의 실제 초기 물리 상태 배열 차이는 0.
  참조에서 채우는 초기 관측 이력도 교체된다. W1은 기존 참조 표현에 의존하며,
  새 보행 실패를 속도나 LLP 문제로 단정할 수 없다. 발 접촉 등 다른 원인을 배제한 것은 아니다.
- 기존 100초 800표본에서 고정 오프셋을 맞춰도 오른발 회전 잔여 RMS는 36.8°.
  고정 회전 하나만 추가해 호환된다고 가정하지 말 것. 기존 A/B 토크 결과·실패 기록은 유지.
- `evaluate_native_wearer.py`에 실제 초기 상태 저장·`clip_end_low_pelvis` 구분 추가.
  과거 raw 결과는 수정하지 않고 분석에서 클립 종료 상태를 구분했다. CPU 3테스트 PASS.
- 결과 폴더 `results/native_walk_validation_20261013/`: 기존 걷기 2/새 느림·보통 16/참조 교체 8/
  빠른 후보 6조건의 summary·trace·로그·SHA256, `validation_summary.json`, 초기 상태 대조 보존.
  이전 A443 첫 2조건도 기존 폴더의 `w1_first_probe/`에 완료. 실행 프로세스는 정상 종료했다.
- **다음:** 기존 참조 생성 규약과 W1 관측 대응을 복원하고 발 접촉을 보완한다.
  필요하면 좌표가 일치하는 실제 보행으로 별도 착용자를 학습/검증한다. 기존 W1은 보존.
  무보조 추종이 확보되기 전 새 데이터로 LLP 성능을 판정하거나 추가학습하지 않는다.
  새로운 착용자/LLP 학습, stochastic/mean 출력 비교는 이번에 시작하지 않았다.

## 2026-10-12 — 실제 보행 전신 변환 후보 생성, 당시 GPU 검증 미완료 (이력)

§2-Y 순서에 따라 **느림·보통 4쌍과 빠른 후보 3개 구간의 SOMA23→suit 전신 변환**을 실행했다.
새 학습은 시작하지 않았다. 상세 NEXT_SESSION **§2-Z**, 단일 파일
[실제 보행 전신 변환 보고서](walking_retarget_report.html).

- 원본 40 Hz·프레임 수·골반 수평 경로/속도 유지. 전신 회전을 suit 관절한계 안에서 맞춘 뒤
  28몸체/27DOF의 위치·회전·속도를 FK로 재계산했다. 관절한계 위반 0, CPU 테스트 3개 통과.
  독립 MuJoCo와 몸체 위치 차이 0.01mm 미만. 원본과 기존 모션·모델은 변경하지 않았다.
- **학습용으로 확정한 데이터가 아니다.** 일정한 골반 높이 보정만으로 지면 침투를 없앴다.
  낮은 쪽 발 높이 95백분위는 느림·보통 약20–48mm, 빠른 후보37–65mm.
  접촉·지지발 높이/미끄러짐을 더 검토해야 한다. 접촉 표식은 힘 측정이 아니다.
- 빠른 4초 창 3개: A417 1.520, A006 1.766, A005 1.907m/s. 정지/회전 구간 제외.
  빠른 걷기인지 달리기인지 확정하지 않았다. HLP 원본 자료 중복에 대한 기존 주의는 유지한다.
- 기존 W1 참조의 저장 몸체 자세와 동일 관절각의 suit FK가 불일치함을 계측했다.
  오른발 위치 차이 평균 약243mm, 회전 RMS 약46°(처음10초 중40표본).
  이는 **두 참조 표현의 차이**이며 실제 W1 추종 오차가 아니다. 기존 A/B 결과를 무효화하거나
  느린 조건 실패의 원인으로 단정하지 않는다. 새 참조의 입력 분포 차이를 검토할 근거다.
- `evaluate_native_wearer.py`: 보조0·고정W1·원본 클립 내부 평가 준비. 샌드박스 실행은
  GPU 접근 오류(`No supported gpu backend found`)로 실패했다. 동일 명령의 샌드박스 밖
  실행을 도구로 요청했으나 **인계 기록 시 응답/성공 결과 없음**. 승인창 표시나 승인 거절을
  추정하지 않는다. GPU 계산 자체는 별도 허용된 확인 명령에서 정상이다.
- 작업root `results/real_walk_retarget_20261012_224957/`. 시작 시71개 파일 백업,
  `native_pairs_v1/suit_native_walk.pt`, `fast_windows_v1/suit_native_walk.pt`, 각 audit·해시·원본,
  `source_snapshot/`, `verification.json` 보존. 무작위/평균 출력 비교와 새 PPO 학습은 미실행.
- 다음: GPU 실행 요청의 실제 상태부터 확인한 뒤 무보조 W1 기존걷기/새8클립을 비교한다.
  발 접촉을 보완하고, 추종이 실패하면 데이터 변환과 착용자 학습 범위를 먼저 검토한다.
  기존 .90배 추가학습이나 달리기 확대를 자동 실행하지 않는다.

## 2026-10-12 — 실제 속도별 보행 데이터부터 점검하도록 순서 변경 (이력)

사용자가 시간변환을 실제 느린 보행과 구분하고, **“실제 느린·빠른 보행 데이터부터 확인하고
추가학습 방향 재검토”**를 선택했다. 아래 §2-X의 0.90배 추가학습을 자동 실행하지 않는다.
새 무작위/평균 출력 비교와 추가학습은 아직 실행하지 않았다. 기존 결과·실패 판정은 보존한다.

- 기존 .85–1.15배는 **동일 전신 모션의 재생 속도**다. 몸체와 다리 속도를 함께 바꾸되 참조
  보폭·발 높이·관절각 경로·접촉 순서/주기 내 비율은 유지했다. 7배율 파일을 실제 비교해 확인했다.
  실제 속도에 맞춰 보폭과 자세를 바꾼 보행 일반의 성공/실패로 읽지 않는다.
- 기존 suit walk의 10초 평균 이동 속도는 1.265–1.306 m/s로 범위가 좁다.
- 로컬 `data/seed/`에서 시간변환하지 않은 BONES-SEED 후보 32개의 원본 BVH와 전신 모션을 확인했다.
  같은 사람 표기 A443의 보통→느림은 이동 속도 1.408→0.694 m/s,
  걸음 빈도 추정 105.3→79.5걸음/분, 한 걸음당 이동거리 추정 .805→.521 m.
  관절각 경로도 다르다. 거리/빈도는 hip 주기 기반 추정이며 착지점 보폭 실측이 아니다.
- 빠른 후보는 이름 충돌·정지/복귀·양발 비접촉 표식이 있어 추가 선별이 필요하다.
  느림/보통 명시 속도군과 빠름 명시 속도군의 사람 표기도 달라 인물 차이가 섞인다.
- 후보는 SOMA23(23몸체/66DOF), 기존 suit는 28몸체/27DOF다. 전신 변환과 무보조 착용자 추종 검증이
  먼저다. 32개 모두 기존 HLP 원본 자료 목록에 있어 시스템 전체 미사용 평가라고 부르지 않는다.
- 보고서: [보행 데이터 점검과 다음 학습 방향](walking_data_audit.html).
  기존 [학습 결과 보고서](walking_finetune_report.html)도 재생 속도임을 명시했다.
  평가 표 45행의 수치는 동일하다. 세부 근거와 다음 순서: NEXT_SESSION **§2-Y**.
- 작업 폴더 `results/llp_slow_walk_audit_20261011_235111/`. 시작 시 미커밋/미추적 70개 파일 백업.
  기존 코드·모델·실험 결과를 수정하지 않았으며 새 학습 서비스도 시작하지 않았다.

## 2026-10-11 — 새 대화에서 시작할 작업 (이력; 다음 순서는 위 인계로 수정)

걷기 학습 결과와 HTML 보고서 수정을 완료했다. **다음 실험은 아직 시작하지 않았다.**
인계 준비 시 실행 중인 `aisbc-*` 사용자 서비스는 없었다. 기존 미커밋 변경과 결과는 보존한다.

다음 작업은 **느린 걷기의 실패가 학습 과정 때문인지, 입력 정보의 한계 때문인지 구분하는 것**이다.
순서는 NEXT_SESSION **§2-X**에 기록했다. 먼저 학습의 무작위 출력과 평가의 평균 출력을 비교한다.
그다음 점검 결과에 따라 0.90배 걷기만 짧게 추가학습한다. 시작 후보는
**부담 비율 반영 400회 모델** (`results/llp_L13R_walk_continue/epoch_400.ckpt`)이다.
기본 속도, 빠른 걷기, 허리와 무릎 부담도 함께 평가한다. 걷기 문제를 해결하기 전에 달리기로 확대하지 않는다.

사용자는 보고서만 읽을 수 있다. 대화에서 설명한 약어도 보고서에서 다시 설명해야 한다.
**AI_SBC_FRAMEWORK §10의 보고서 작성 원칙**을 적용한다. 표와 그래프에도 설명 이름을 사용한다.
수정 보고서: [걷기 보조 모델의 학습 결과](walking_finetune_report.html).

## 2026-10-11 — 전신 걷기 추가학습·독립 검증·속도 확대 결과 (최신)

사용자가 승인한 **L9→전신 걷기 추가학습→독립 비악화 검증→걷기 조건 확대**를 실제 수행했다.
L13 200 epoch, L13R 200+200 epoch, 속도 혼합 L14 200 epoch와 각 독립 평가 완료.
**정상 걷기는 크게 개선됐지만 느린 걷기는 실패했다. 달리기 확대 조건은 미충족**이다.
최종 목표인 걷기·달리기·계단·험지 general model을 얻었다고 해석하지 않는다.

- [최종 단일 파일 HTML](walking_finetune_report.html): 이전 실험의 실패 이유, 전체 phase,
  학습 계보, 새 시드·구간·속도 평가와 전신 부담을 함께 담았다. 이미지도 내장했다.
- 고정 후보 `results/llp_L14_walk_tempo/epoch_200.ckpt`: 정상 seed42/start0에서 hip τ²
  **−69.45%**, 20–40% RMS 변화 **−2.45 N·m**, 70–90% **+0.47 N·m** (L9는 +6.43).
  기존 L1·L9 **걷기 개발 기준 전체 통과**. 다만 80–90%에는 +2.24 N·m 악화가 남는다.
- 새 평가 seed73/start0, seed42/start60, seed97/start400에서 τ² **−65.61~−69.27%**,
  70–90% +0.46/+1.54/+2.72 N·m. 세 조건 모두 같은 조건의 L9 대비 검증 통과, 40초 생존.
- 0.90배 start0/start60: 70–90% **+9.71/+10.87 N·m**, body **+3.38/+5.36 mm**로 실패.
  학습에 포함된 속도이므로 미사용 속도 외삽만의 문제로 설명할 수 없다.
  미사용 .85/.95도 body 기준 미달, 1.05/1.15는 양호. 모든 평가 40초 생존.
- L14 정상 걷기 전신 계측: hip −69.45%지만 **허리 +34.27%, 무릎 +5.22%**.
  다리 10개 회전관절 τ² 합계 −22.31%, 전체 23개 −3.67%. 대사량 감소 지표가 아니다.
- 같은 짧은 hip 관측창과 동결 HLP를 유지해도 개선됐다. 앞선 계측의 무릎·골반 결합과
  전신 추가학습 효과는 진자 전이의 한계를 뒷받침한다. HLP 20 ms 미래각은 느린 조건에서도
  RMSE 약 0.63°였다. 관측 길이나 HLP만의 문제로 단정하지 않는다.
- 남은 우선순위: 느린 조건의 비악화와 허리·무릎 부담 이동. 단순 epoch 연장이나 달리기 확장부터
  하지 말 것. 조건별 목적 충돌과 학습 환경 내 실패를 먼저 읽을 것: NEXT_SESSION **§2-W 최종 결과**,
  `results/llp_L14_train_domain_audit_20261011/summary.json`.
- 최종 평가 `results/llp_L14_final_validation_20261011/`, 재현 manifest
  `results/llp_walk_report_20261011/final_manifest.json`. 기존 54개 미커밋/미추적 파일 백업 및
  모든 기존 결과·실패 로그 보존. L12 진자 recipe는 미학습. 소스 checkpoint도 보존했다.

## 2026-10-10 — 걷기 전신 계측·제동 차단 진단 실제 완료 (이력)

앞 세션은 분석/준비에서 멈췄으나, 이번에는 **GPU에서 전신계측과40초 비교까지 완료**했다.
상세 NEXT_SESSION §2-U, [전신 진단 HTML](walking_fullbody_diagnostic.html) (단일파일).

- 기존 L9@2000/W1/실제HLP/걷기 seed42: 계측없는8초×2 → 계측8초 → L9 40초 →
  L9의참조phase70–90 보조만0으로한40초. 마지막조건은 특권 진단이며 배포정책 아님.
- **70–90 해당 발 접촉0, 무릎이빠르게펴지는 구간**으로실측.70–80 A의관성식에서
  hip자체−95.74Nm, 무릎결합+75.88, 골반결합+16.49 등이상쇄돼 착용자평균+4.73.
  단순진자에없는 큰결합항을 계측. 좌표별 항분해이며 무릎고정 인과실험은아님.
- 80–90 A착용자−3.68, 원래L9보조−16.12, B착용자+13.71Nm: 착용자가제동을상쇄하는방향.
- 보조차단후 전체τ² **−25.88→−30.11%**,70–90 RMS증가 **+6.43→+2.14Nm**.
  80–90 B RMS **22.35→17.65Nm** (A≈16.6). 두실행40초생존·몸체오차/속도악화없음.
  40–50절감폭은약2.1Nm줄어드는교환이남고,70–80도완전해결은아님. 전구간표보존.
- 전체상태/PD목표100Hz, 동역학/접촉400Hz를별도기록. 정책입력변경없음.
  계측변동은기존반복산포안. CPU별도관성/편향검증, 토크창정렬검증 통과. 기존14테스트PASS.
- MJWarp sparse qfrc_constraint는수렴world의scratch가0일수있어 최종활성제약Jᵀf로재구성.
  솔버잔차RMS≈0.33Nm(<1%), 최대≈16.5Nm를별도공개. 모델/패키지수정없음.
- root `results/walk_fullbody_diagnostic_20261010/`, source_snapshot_40s, comparison.json/html,
  sequence_status.json=complete. 서비스정상종료·새학습없음. L12는여전히미학습.
- 기존49파일백업 preexisting_232135, 기존결과/미커밋보존. NVML오류와별개로CUDA계산정상,
  승인대기없음. 다음은 L9를기준으로 무릎/전신결합을배울수있는플랜트·비악화목적 검토.

## 2026-10-09 CLI 재개 — 걷기 원인 진단, 관측창/진자 한계 재검토 (이력)

**사용자가 CLI에서 재개를 지시했다. 아래 이전 일시정지 기록은 이력이다.** 걷기 우선,
근거 없는 장시간 반복 금지. L11 재학습 없음, 새 실험/서비스는 아직 시작하지 않았다.

- [걷기 진단 HTML](walking_diagnostic.html): 토크 구간, 실제 과거·미래 관측,
  정책 CPU 재생 및 창 길이 정보 진단을 한 파일에 내장. HTML 하나만 다운로드해 Chrome에서 열면 된다.
  기존 l6_l7 보고서·모든 원본 결과는 보존. 상세 **NEXT_SESSION §2-T**.
- 20–30%: L6 RMS 22.41→38.90 N·m, L1 22.39→22.61. 새 정책의 늦은 토크 방향 전환 문제.
  70–90%: 무보조 RMS 약17–18로 0은 아님. L1의 보조16–20도 이 구간에선 상당한 크기.
- **L9@2000은 걷기 후보로 보존:** 전체 τ² −25.44%(L1 −19.32), 최악 RMS 증가7.05(L1 8.83).
  다만 20–40%는 −1.706→+2.257 N·m로 악화. 기존 합격 판정·실패 결과를 지우지 않는다.
- **70–90% HLP는 대체로 계속 전진하는 미래를 정확히 예측한다.** 실제 미래각으로 교체해도
  L9 평균 제동 −14.63/−16.48 → −15.37/−16.13 N·m로 남음. HLP 정확도만의 문제 아님.
  과거를 등속으로 바꿔도 L9 제동이 커져 “감속 이력 하나 때문”이라는 단정도 금지.
- 사용자 가설: 짧은 moving window가 전체 보행 맥락/이완 구간을 못 보는가, 진자 자체 한계인가.
  주기1.12초, 과거창22%·미래창1.8%. 같은 기록 시간분할 진단에서250ms에도 phase 정보가 상당하고
  1초 확장만으로 개선 없음. 일반화/LLP의 정보 사용 입증은 아님. **구간을 구별할 정보 부족**과
  **진자에서 배운 필요 토크가 전신과 다름**을 분리해서 판단할 것. 현재는 후자를 더 우선할 가설로 본다.
- 같은 W1 hip 궤적의 진자 요구 토크도 비교. 70–80% 평균 전신 +4.72 vs 중간 관성 진자 −37.37 N·m.
  그러나 80–90은 진자도 +3.07이므로 지속 제동 전부를 관성 하나로 설명할 수 없음.
  외란 없는 다른 플랜트의 역동역학 비교이며 W1 내부 토크 분해/정책 성능 결과는 아님.
  MuJoCo CPU 60상태 검증. 상세 NEXT_SESSION §2-T 및 HTML 마지막 표.
- L12 paired non-degradation tracking 코드/검증/걷기 우선 runner 준비, 관련 CPU25테스트 PASS.
  실제 물리·4096메모리·PPO·W1 검증은 미실행. 모델/관측/나머지 보상은 L11과 같음을 config로 검증.
  **보상 수정만으로 플랜트 한계까지 해결된다고 가정해 바로 학습을 걸지 말 것.**
- 기본 sandbox는 GPU/systemd 접근 실패했으나 이미 승인된 명령을 sandbox 밖에서 실행해
  CUDA true/RTX5090 확인. 사용자 터미널도 정상. GPU 고장 아님. 신규 검증 명령은 도구 응답 없이
  미실행이어서 취소. 사용자 화면에 승인창이 떴다는 증거 없음. 실행 접근 상태는 §2-T 참고.
- 기존 미커밋37개 파일 백업·해시: `results/cli_walk_diagnostic_20261009/preexisting_230926/`.
  이번 추가 변경도 미커밋으로 보존. 앱 자동화는 임의로 켜지 않았다.

## 2026-10-09 18:40 KST — L11 완료 후 사용자 요청으로 일시정지, CLI 인수인계

**이 기록이 아래 실행 중/재개 기록보다 우선한다. 사용자가 CLI에서 재개를 지시할 때까지 새 작업을 시작하지 말 것.**

- 사용자 요청: "지금 하는 작업을 잠시 멈추고 cli에서 다시 하려고 해."
- L11은 6000 epoch 및 2k/4k/6k × 걷기·달리기 평가를 모두 완료했다. 학습을 다시 시작할 필요 없음.
  18:39 KST 확인: 실행 중 aisbc 서비스 없음, GPU compute 프로세스 없음. 앱 자동화 l6-l7은 PAUSED 유지.
  이 대화의 활성 목표도 사용자 요청으로 일시정지한다. 완료된 실험 status.json은 complete 상태를 보존한다.
- **L11 모든 checkpoint 실패:** 걷기 τ²가 무보조보다 +141.36/+84.31/+68.02%,
  달리기는 각각 7.9/6.3/8.9초 낙상. 추론 gain1/최종±100 N·m. 자세한 수치·검증은 NEXT_SESSION §2-S.
- 분석 파일: `results/llp_L11_opposition_sequence/analysis_L11.json`, `posttrain_reward_diagnostic.json`.
  보고서: `results/llp_L11_opposition_sequence/comparison.html` 및 `docs/l6_l7_torque_report.html`.
  기존 기준에 따른 L1×0.5 + L6@6000/L9@2000/L8@4000만 표시하며, 실패 원본은 모두 보존했다.
- 합성 진자에서 추종 보상 증가가 부담 절감 보상 증가보다 큰 현상을 확인했다. 독립 제어기 일치 검사는 통과.
  W1 실패의 인과적 원인을 입증한 것은 아니다. 일반화 성능 개선을 주장하지 않는다.
- **다음 가설은 검토만 했으며 구현·학습하지 않았다. L12는 아직 없다.** 동일 플랜트·목표·외란의
  무보조 진자를 함께 돌려, 무보조보다 추종을 악화시킬 때만 추종 비용을 주는 방식을 우선 검토한다.
  착용자가 원래 하던 움직임보다 목표를 더 정확하게 맞추는 것만으로 추가 보상을 받는 유인을 줄이려는 가설이다.
  다음 에이전트는 설계 타당성·쌍별 RNG/부분리셋/종료 처리·0보조 일치·메모리부터 검증할 것.
- CLI 재개 시 이 문서와 NEXT_SESSION §2-S를 먼저 읽고 다음 가설을 구현·검증한 후 새 서비스 하나를 시작한다.
  예전 L11 시작 명령을 새 실험으로 착각해 실행하지 말 것. 앱 자동화를 임의로 다시 켜지 않는다.
- 모든 미커밋 변경·체크포인트·원본 결과를 보존했다. 하드웨어 보류, actor 입력/크기 유지,
  gain1/±100 N·m, phase 포함 선별 기준, 추가 seed/미사용 조건 검증 방침을 유지한다.

## 2026-10-09 17:21 KST — 사용자 목표 재활성화에 따라 L11 재개 (이력)

- 사용자가 개선 목표를 active로 전환하고 계속 진행하도록 요청해 이전 일시정지를 해제했다.
- 기존 서비스·GPU 학습 프로세스가 없음을 확인했다. L11 last.ckpt epoch1000의 SHA256이
  pause_state.json과 일치하고, 학습 시작 당시 source_snapshot의21개 파일도 모두 불변이었다.
- 같은 시작 스크립트로 **17:20:53 KST** `aisbc-llp-L11-opposition.service`를 재개했다.
  실제 체크포인트/환경 로드 및 epoch1000→1018 증가, active/running·재시작0·GPU 사용 확인.
- 학습 설정은 그대로이며 6000epoch까지 학습한 뒤 2k/4k/6k 걷기·달리기 평가를 실행한다.
  현재 상태 `results/llp_L11_opposition_sequence/status.json`, 재개 근거 `resume_state.json`.
- 앱 heartbeat l6-l7은 중복 실행 방지를 위해 PAUSED 유지. 활성 목표의 후속 작업에서
  동일 서비스 상태를 확인하며 이어간다. 정상 학습 중 설정 변경·새 학습·중복 실행 금지.
- 아래 일시정지 기록은 이력이다. 현재는 **실제 학습 중**이며 아직 L11 평가 결과는 없다.

## 2026-10-08 15:53 KST — 사용자 요청으로 일시정지, CLI 인수인계

**현재 학습은 멈춰 있다. 사용자가 CLI에서 재개를 요청하기 전에는 자동으로 시작하지 말 것.**

- 사용자 요청: "지금 하는 작업을 잠시 멈추고 cli에서 다시 하려고 해."
- `aisbc-llp-L11-opposition.service` 수동 stop 완료: inactive/dead, MainPID0, Result=success.
  GPU compute 프로세스 없음 확인. 오류에 의한 종료가 아니다.
- `results/llp_L11_opposition/last.ckpt` **epoch1000**, 환경 `env_null.ckpt` 로드 검증 완료.
  로그의 마지막 epoch999는 0-based 학습 루프이며 저장된 다음 epoch1000에서 재개한다.
- 자동 추적 `l6-l7`은 **PAUSED**. CLI 작업 중 앱이 중복 실행하지 않도록 유지한다.
- 상태/재개 근거 `results/llp_L11_opposition_sequence/{status,pause_state}.json`.
- 사용자 재개 지시를 받으면 아래 명령을 레포 루트에서 실행한다. 기존 last.ckpt·resolved config를
  자동으로 읽어 L11을 이어서 학습하고, 6k 이후 2k/4k/6k 걷기·달리기 평가까지 순차 실행한다.
  새 실험 이름으로 바꾸거나 --checkpoint로 다른 모델을 지정하지 않는다.

```bash
bash examples/experiments/assist_pendulum/start_mixed_feedforward_sequence.sh
```

재개 후 실제 epoch 증가·GPU·서비스 상태를 확인할 것. 성능 개선 완료로 해석하지 말 것.
아래는 중단 전 이력이다. 실험 근거와 기존 결과는 NEXT_SESSION §2-R, 상세 제약은 이 문서의
운영 규칙을 따른다. 하드웨어 보류, gain1/±100 N·m, 기준선+상위3개 보고, 미달 원본 보존 유지.
모든 변경은 현재 작업 디렉터리에 있으며 이번 작업에서 커밋하지 않았다. 기존 변경을 덮어쓰지 말 것.

## 2026-10-08 15:41 KST — L10 실패, L11 착용자 제어 방식 혼합 학습 중

- L10은 15:13:23 완료. 2k 걷기 70–90% 착용자 RMS +13.11 N·m(기준선 +6.96),
  4k 달리기 11.4초 낙상. 전 checkpoint 개발 기준 미달로 상위3개에 진입하지 못함.
  다른5개 평가의 40초 완주는 meta와4000표본으로 확인. gain1/최종±100 N·m.
- 첫 평가 중 SIGILL/SIGSEGV 두 번 자동 재시작 후 회복. 원인 확정 불가, 결과 파일 검증 완료.
- 보고서 `results/llp_L10_opposition_sequence/comparison.html`, `docs/l6_l7_torque_report.html`:
  L1×0.5 + L6@6000/L9@2000/L8@4000만 표시. L10 포함 모든 실패 원본 보존.
- 다음 L11은 L9 기준에서 착용자 속도 FF 유/무를 env별 50% 확률로 혼합.
  근거: 같은 합성 플랜트/목표에서 L9@2000의 τ² 절감이 FF유71.9%→FF무19.0%로 하락,
  반대부호 작용44.3→70.4%. W1 실패 원인 입증은 아니며 해당 제어방식 차이를 검증한다.
  L10 임피던스 확장은 적용하지 않음. 모델/관측/보상/추론배율 불변.
- 관련7개 테스트·4epoch 소규모 GPU학습·실제 혼합 착용자 주입/부분리셋 확인 통과.
- `aisbc-llp-L11-opposition.service` **15:39:25 시작**, 15:40:35 실제 epoch61,
  active/running·재시작0·GPU 프로세스7222 MiB 확인. 6k 후 같은6회평가 자동 실행.
  상태 `results/llp_L11_opposition_sequence/status.json`, 로그 `train.log`.
  시작 스크립트 `examples/experiments/assist_pendulum/start_mixed_feedforward_sequence.sh`.
- 상세 결과·진단 한계·검증은 NEXT_SESSION §2-R. 다음 자동 추적 L11, L10 완료 중복 보고 금지.
  아직 일반화 개선 입증 없음. 하드웨어 작업 보류 유지.

## 2026-10-08 13:49 KST — L9 완료·달리기 미달, L10 착용자 임피던스 확장 학습 중

- L9 6천 epoch 및 6회 W1 평가가 13:31:51 정상 완료, 재시작 0. 모두 낙상 없이 40초 완주.
- 보고서에 선정된 L9@2000: 전체 τ² 걷기 −25.44% / 달리기 −17.47%,
  최악 phase RMS 증가 +7.05 / +17.97 N·m. L1의 +8.83/+12.13 대비 걷기는 개선·달리기는 악화.
  걷기 20–40%도 +2.26 N·m로 아직 방해. **모든 L9 checkpoint가 사전 개발 기준 미달.**
- 보고서 `results/llp_L9_opposition_sequence/comparison.html`와 `docs/l6_l7_torque_report.html`:
  L1×0.5 + 상위 최대 3개(L6@6000, L9@2000, L8@4000). 원본 전수 결과 보존.
- 다음 가설 L10: L9 보상/모델/추론을 유지하고, 합성 착용자 고유진동수 범위 12–24 → 6–24 rad/s.
  가벼운 진자의 지나친 약화를 막는 착용자 Kp 하한 60 N·m/rad, 최종 Kp로 감쇠비 0.3–0.8 유지.
  보조 토크 상한 변경 아님: 추론 gain 1 / 최종 ±100 N·m 유지. W1 보행 학습 데이터 미사용.
- 근거: 기존 진자는 I≥3 kg·m²에서 Kp≥432여서 W1 hip 게인 200/20 근방의 약한 조합을 못 봄.
  이는 합성 플랜트의 분포 공백이며 W1 전신 동역학의 동정이나 실패 원인 입증은 아님.
- `aisbc-llp-L10-opposition.service` **13:48:35 시작**, epoch 0→6 실제 증가 확인,
  active/running, 재시작 0. 상태 `results/llp_L10_opposition_sequence/status.json`, 로그 `train.log`.
  시작 스크립트 `examples/experiments/assist_pendulum/start_soft_impedance_sequence.sh`.
- 전체 설정 비교, 적용된 Newton 게인 점검, 4 epoch GPU smoke 및 관련 7개 테스트 통과.
  자세한 L9 결과·L10 가설·무보조 점검 한계는 NEXT_SESSION §2-Q. 다음 추적 대상 L10.

## 2026-10-08 12:07 KST — L8 완료·phase 기준 미달, L9 실제 학습 중

- L8 서비스 정상 종료(11:51 KST, 재시작 0), 6천 epoch 및 2k/4k/6k × 걷기/달리기 평가 완료.
  전부 4000 표본/39.99 s, `survived_s=null`: 낙상 없이 40초 완주 확인.
- L8@4000 전체 착용자 τ² 걷기 −23.74%, 달리기 −28.54%로 L1×0.5보다 개선.
  그러나 걷기 20–40% RMS +13.44 N·m(기준선 −1.71), 달리기 70–90% +4.43(기준선 −1.07).
  **L8 모든 체크포인트 개발 phase 기준 미달. 평균 개선만으로 성공 판정하지 않는다.**
- 보고서: `results/llp_L8_opposition_sequence/comparison.html`, `docs/l6_l7_torque_report.html`.
  기준선 + 최악 phase 기준 상위 3개(L6@6000, L8@4000, L8@2000)만 표시. 전수 결과 보존.
- 다음 가설 L9: 관절별 반대 토크 비용을 `log1p(relu(−human*assist)/100)`으로 변경.
  작은 토크의 반복적 방해에 상대적으로 무게를 주며, 어느 쪽 반대 토크가 커져도 벌점이 줄지 않는다.
  합성 진자 229,752 유효 env-step으로 평균 벌점을 L8과 맞춘 가중치 −0.021958557930324075.
  reward 외 플랜트·actor·나머지 보상은 동일, 추론 배율 1 / 최종 ±100 N·m 유지.
- `aisbc-llp-L9-opposition.service` **12:06:57 KST 시작**, 실제 epoch 0 → 18 증가 확인,
  active/running, 재시작 0. 6k 학습 후 같은 6회 평가. L9 결과는 아직 없다.
  상태 `results/llp_L9_opposition_sequence/status.json`, 로그 `train.log`.
  시작 스크립트 `examples/experiments/assist_pendulum/start_relative_opposition_sequence.sh`.
- 자세한 분석·검증·가설 한계는 NEXT_SESSION §2-P. 일반화 검증·하드웨어 작업 완료로 해석하지 말 것.

## 2026-10-08 10:26 KST — GPU 접근 확인, L8 실제 학습 시작

- 현재 Full access 실행 환경에서 RTX 5090, CUDA 장치 1개, 사용자 systemd 접근 정상 확인.
- `start_opposition_sequence.sh` 실행 성공. `aisbc-llp-L8-opposition.service` active/running,
  재시작 0회. 실제 학습 로그에서 epoch 0 → 33 증가 및 학습 프로세스 GPU 메모리 7,208 MiB 확인.
- L8: L6에 관절별 반대 토크 비용 −1e-4만 추가, 새 학습 6,000 epoch.
  추론 배율 1 / 최종 ±100 N·m 유지. 학습 후 저장된 2k/4k/6k를 걷기·달리기 각각 40초 평가.
- 상태 `results/llp_L8_opposition_sequence/status.json`, 상세 로그 같은 폴더 `train.log`.
  완료 시 `comparison.html` 및 이 문서 아래 연결된 phase 보고서를 갱신한다.
- 기존 알림 자동화 `l6-l7`을 **LLP phase 개선 실험 추적**으로 갱신·활성화.
  완료 결과를 분석하고 필요하면 검증 가능한 다음 가설 하나를 순차 실행하도록 설정.
- 아래 접근 차단 기록은 이전 실행 환경의 이력이다. **현재 학습은 실행 중이며 성능 개선은 아직 미확인.**

## 2026-10-08 — 평균의 함정 재평가, L8 준비 / 이전 실행 접근 차단

세 번째 연속 확인에서도 CUDA 장치 0개·사용자 systemd bus 접근 거부. 보이는 학습 프로세스와 L8 체크포인트 없음. 분석/구현/CPU 검증 이후 필수 다음 단계인 실제 학습·평가를 진행할 수 없어 지속 개선 목표를 `blocked`로 전환. GPU와 사용자 서비스 접근이 복구되면 L8 실행 스크립트부터 재개할 것. 목표 달성/성능 개선으로 해석하지 말 것.

- 사용자 요청: 좋은 결과까지 분석·개선 반복. 비교는 L1×0.5 + 상위 최대 3개만.
- L6@6000의 걷기에서 20–40%, 70–90% 착용자 RMS가 각각 +13.7, +9.9 N·m 증가.
  전체 τ² −25.3%만으로 L1보다 좋은 정책이라고 판정할 수 없음.
- L6는 power+τ², L7은 τ² 중심. 보고서는 최악 phase 악화를 우선해 후보 선정하도록 수정.
  현재 완주·양쪽 전체 절감을 만족하는 L6@6000/@4000만 표시(L1 별도).
- 개발 판정 `phase_screen.py`: 전체·최악 phase·문제 구간·추종·동작 변화를 각각 확인. 현재 L6/L7 모두 미달. 후보 순위와 합격을 구분.
- L8: L6에 관절별 `relu(−tau_agent*tau_assist)` 비용만 추가하는 가설.
  코드·설정 생성·CPU 테스트 통과. 자세한 내용 NEXT_SESSION §2-O.
- **당시 L8 미학습**: 당시 restricted 실행 환경에서 CUDA가 안 보이고 사용자 systemd bus 접근 거부.
  준비 상태 `results/llp_L8_opposition_sequence/readiness.json`, 접근 복구 후 실행 스크립트
  `examples/experiments/assist_pendulum/start_opposition_sequence.sh` 사용.

## 2026-10-07 결과·보고서 갱신

- L6/L7 6천 epoch 학습·12회 평가 완료(17:21 KST), 서비스 정상 종료. 완료 알림 자동화 중지.
- L6@6000: τ² 걷기 −25.3%, 달리기 −19.3%, 둘 다 40초 완주. L1×0.5보다 걷기는 개선,
  달리기는 부족. L7@6000: +8.4/+51.1%로 부담 증가. 학습 seed 1개·개발 모션 평가로 해석.
- 사용자 요청으로 [phase·토크 상세 보고서](l6_l7_torque_report.html) 추가.
  기존 `results/llp_wide100_sequence/comparison.html`도 같은 새 보고서로 교체.
  L1 기준선과 L6/L7의 2k/4k/6k, 걷기/달리기 선택 가능.
  `τ_agent A/B`, `τ_assist A/B`, `τ_total`을 각각 분리해 phase·시간 축으로 표시하고
  phase별 RMS 감소량(N·m), τ² 변화율(%), CSV 저장 제공.
- 생성기 `examples/experiments/assist_skeleton/report_torque_html.py`.
  토크 정렬은 agent[i] + assist[i−1], 시각·phase는 해당 창의 중점.
  phase 0%는 mocap hip 최대 굴곡이며 접촉 검출값 아님.
  원본 14개 평가의 phase 기여 합·합산 토크 검증과 JS 선택/CSV 검증 통과.
  브라우저 로컬 파일 접근 차단으로 화면 렌더링 미검증.

## 2026-10-07 추가 — L6/L7 순차 학습 시작 (완료 전 기록)

- 사용자 승인에 따라 **시뮬레이션 고도화만 진행**, 하드웨어 확인은 사용자 담당.
- `mlp_wide.py`: `40 × raw action`으로 요청 토크를 만들고 최종 ±100 N·m에서 제한.
  tanh와 추론 전용 0.5배 보정 없음. 모델 크기·관측 유지.
- L6는 기존 power+τ² 보상, L7은 power 항만 제거. 두 조건 모두 새로 학습한다.
- 진자의 보조+착용자 FF+외란 합을 다시 ±150으로 제한하던 경로는 새 실험에서 해제.
  각 소스의 개별 제한은 유지하며 기존 L1 경로의 기본 동작은 보존했다.
  **L1 대비에는 이 수정도 포함되므로 한계만 바꾼 실험으로 해석하지 말 것.**
- 검증: 테스트 12개, 실제 ±160 요청→±100 적용 및 주입 오차 0,
  독립 제어기 등가성 PASS. 축소된 두 학습·네 평가 순차 실행도 완료.
- 본 실행 서비스 **`aisbc-llp-wide100-sequence.service`**, L6부터 시작.
  L6 6,000 epoch → 저장된 2k/4k/6k W1 걷기·달리기 40초 평가 → L7 동일 순서.
  두 학습을 동시에 실행하지 않는다. 추론 배율 1, H1/W1 고정.
- 현재 상태 `results/llp_wide100_sequence/status.json`; 실행 로그
  `results/_pids/llp_wide100_sequence.log`, 상세 학습 로그 `results/llp_wide100_sequence/train_L6.log`.
  결과는 같은 디렉터리의 `comparison.json`/`comparison.html`에 평가마다 갱신한다.
  **성능 개선은 아직 확인 전.** 자세한 설정·검증·판단 기준은 NEXT_SESSION §2-N.

## 2026-10-01 추가 — Codex 작업 재개

- 사용자 결정: **일반화 개선을 진행한다. SBC 작동 확인은 사용자가 별도로 수행하며 하드웨어 작업은 보류.**
- L1 독립 제어기 검증 수정·통과: 검사 스크립트의 옛 40 Hz HLP 선택을 H1 100 Hz로 통일.
  4 env × 600 step, 부분 reset 4회에서 관측·raw action·적용 토크 차이 0.
  HLP 입력 버퍼 차이 ≤1.87e-9 rad, 난수를 맞춘 200 step의 예측·action 차이 0.
  이는 **실 HLP로 교체한 진자 env ↔ 독립 제어기**의 검증이며 스켈레톤 전체 등가성은 아니다.
- W1 새 40초 A/B: H1+L1×0.5 걷기 τ² **−19.3%**, 달리기 **−23.7%**, 둘 다 완주.
  [W1 보고서](l1_w1_report.html), [설정·해시·수치](l1_w1_baseline.json). 기존 W0 보고서는 역사 자료.
- **포화율 정정:** 기존 분석은 배율 적용 후 토크를 적용 전 한계와 비교했다.
  기존 W1 L1×0.5 기록의 포화율은 0%가 아니라 **53.6 / 69.3%**(걷기/달리기, 20 N·m의 95% 초과).
  τ² 감소율은 변하지 않는다. L2의 90~97%와 비교할 때 이 수정값을 사용할 것.
- 기존 L1·학습 보상·제어기 출력은 변경하지 않았다. 이 기록 시점에는 추가 배율 없는 새 정책을 학습하지 않았다(10-07 L6/L7 시작).
  다음 학습은 한계 랜덤화만 반복하기보다 **요청 토크와 제한 후 토크를 구분하는 관측·행동 계약**을
  먼저 설계하고, 진자에서 토크 크기 조절이 되는지 확인한 뒤 W1에서 검증한다.
  W1 걷기·달리기는 이미 개발에 반복 사용했으므로 최종 일반화 시험으로 주장하지 않는다.

---

## 1. 무엇을 만드는가

고관절 보조 장치(AI-SBC)의 제어기. **실기기에는 두 모델만 올라간다.**

| 구성 | 하는 일 | 학습 | 위치 |
|---|---|---|---|
| **HLP** (High-Level Predictor) | 지난 100 ms의 고관절 각도·몸통 자세로 **앞으로 0~50 ms의 고관절 굴곡각**(10 ms 격자 6점)을 예측 | 사람 모션 데이터, 지도학습(ManiFlow) | sibling 레포 `~/Projects/ManiFlow_Policy`, 이 레포에서는 `protomotions/maniflow/`로만 접근 |
| **LLP** (Low-Level Policy) | HLP 예측 + hip 각도·각속도 히스토리로 **매 10 ms 좌우 hip 굴곡 보조 토크**를 냄 | 시뮬레이션 PPO | `checkpoints/L*`, 학습 코드 `examples/experiments/assist_pendulum/` |
| **가상 착용자** (W, mimic tracker) | 시뮬 안에서 사람 역할. mocap을 따라 걷고 뛴다. 실기기에서는 진짜 사람 | 시뮬레이션 PPO | `checkpoints/W*`, 학습 코드 `examples/experiments/assist_skeleton/tracker_*.py` |

사용자가 못 박은 목표(원문): **"assist torque를 출력하는 RL모델이 어떤 input이 들어오던간에 올바른 torque assist를
해주는게 목표"**. 특정 착용자·특정 동작에 맞춘 정책이 아니라, 입력이 무엇이든 올바른 보조를 내는 정책이다.
그래서 LLP는 hip-local 최소 관측(각도·각속도·예측)만 쓰고, 착용자나 보행 데이터를 학습에서 보지 않는 쪽으로 왔다.

고정된 범위: **보행 계열(걷기·달리기)만**. 인체 모델은 해부학적 뼈 스켈레톤(`skeleton_torque_suit.xml`).
착용자 특이성(여러 사람 대응)은 뒤로 미룸.

---

## 2. 지금 결론 (2026-10-01)

**배포·평가 후보 = H1 + L1 × 0.5 @ W1.** 걷기 τ² −19.3 %, 달리기 −23.9 %, 낙상 0, 보조 RMS 17 / 19 N·m.
(τ² = 착용자 hip 토크 제곱의 합, 근활성 프록시. 무보조 A 대비 보조 B의 변화. §5 참고.)

| W1 위에서 (40 s A/B) | 걷기 τ² | 달리기 τ² | 비고 |
|---|---|---|---|
| **H1 + L1 × 0.5** | **−19.3 %** | **−23.9 %** | 현재 후보. `--assist-scale 0.5` ≈ 한계 20 N·m |
| H1 + L1 그대로 (40 N·m) | −0.1 % | 4.4 s 낙상 | 사람 범위 착용자에겐 과보조. **쓰지 말 것** |
| L2 (한계 U[10,50] 랜덤화 + 질량∝관성) | −16 % | −24 % | 포화 90~97 %(bang-bang), 구동기 불가 |
| L3 (+ 하중 절반, 반전 점성 브레이크) | −12.5 % | −23.7 % | |
| L4 (L3 + chunk를 실제 각도에 정박) | +0.9 % | −7.8 % | 학습할수록 악화(2k → 4k) |
| L5 (환경이 반전 관성을 대신 냄) | 무효 | 무효 | 정책 출력 ≈ 0 (RMS 7, 상관 0) |
| β 오라클 0.5 (특권 상한) | −35.1 % | −40.8 % | 착용자 토크의 절반을 그대로 내는 규칙. 정책이 아님 |
| 마스크 진단: L1×0.5 + 위상 70~100 % 보조 0 | −15.6 % | −30.3 % | 보행 위상을 아는 특권 진단. 정책이 아님 |

- 옛 개인화 정책(스켈레톤 위에서 추가 학습한 llp_v3~v12)은 τ²로 재면 **달리기를 한 번도 못 도왔다**(v10 +21, v11 +230, v12 +9 %).
- W0(옛 v32, 사람보다 1.7~3.7배 센 착용자) 위의 L1 −30.5 / −25.5 %는 **잣대에서 뺐다**. 사람 범위 착용자 W1이 잣대.
- 2026-09-29 밤 진자 변형 L2~L5 네 개를 돌렸고 모두 L1×0.5 아래 → 전부 중단. 당시 GPU는 비워 두었음. **현재 상태는 위 10-07 기록 참조**.
- **사용자 결정(10-01)**: 현재 성능을 비교 기준으로 보존하고 일반화 개선 진행. 하드웨어 확인은 사용자 담당. 위 추가 기록 참고.
- 2026-09-30에 사용자가 "현재로선 L1 × 0.5가 가장 좋다"는 판단을 확인했다. 단 배율 0.5 자체는 최적값이 아니라 처음 시도한 값이다
  (이전 W1 스냅샷 걷기에서 0.3 < 0.5 한 번 확인, 0.4~0.7 미탐색).

---

## 3. 시스템 구조 (한 화면)

```
 스켈레톤 착용자 W1 (Newton, 물리 400 Hz)         ← 사람 역할. mocap 추종 PPO. hip 한계 200 N·m
      │ 10 ms마다 (env step 50 ms 안의 서브스텝 훅, protomotions/simulator/newton/substep_hook.py)
      ▼
 hip_flexion_r/l 각·각속도 + trunk pitch/roll ──▶ AssistController (protomotions/assist/controller.py)
                                                  ├ HLP H1: 과거 100 ms → 미래 0~50 ms, 10 ms 격자 6점 (3.2~4.9 ms)
                                                  └ LLP L1: 히스토리 250 ms + 미래 창 {0,10,20 ms} + 이전 action 100 ms → tanh × 40 N·m × 배율 (0.36 ms)
      ▲                                                                              │
      └────────── qfrc_applied 주입 (hip flexion r/l) ◀──────────────────────────────┘
 A env = 보조 0, B env = 보조 켬. 같은 mocap·같은 seed로 짝지어 40 s 돌리고 착용자 토크를 비교한다 (= 통합 평가).
```

| 주기 | HLP 10 ms | LLP 10 ms | 물리 2.5 ms (400 Hz, decimation 4) | tracker env 50 ms (관측이 묶여 있어 못 바꿈) |
|---|---|---|---|---|

**LLP가 보는 것**: hip 2ch 각도·각속도 + 히스토리 25 step + HLP 예측 창 {0, 10, 20 ms} + 이전 action 10 step. **못 보는 것**:
착용자 근력·관성·하중·보행 위상·모션 종류. 그래서 같은 mocap을 따르는 두 착용자(W0, W1)에게 같은 토크를 낸다.

**용어 (2026-09-29 사용자 지시)**
- **통합** = HLP·LLP를 각각 따로 학습한 뒤 그대로 붙여 평가·배포. `run_system.py`의 스켈레톤 A/B가 곧 통합 평가. 기본 목표.
- **개인화** = 통합 뒤 특정 착용자 위에서 LLP를 추가 학습(`SkeletonAssistEnv`, `llp_walk.py`; 옛 v3~v12). 지금은 안 씀.

**버전 체계 (2026-09-29)**: 구성 요소 접두어 + 번호, 현재 계보는 1부터. 조합은 "H1+L1 @ W1"처럼 쓴다.

| | 현재 | 옛 이름 | 위치 | 내용 |
|---|---|---|---|---|
| H | **H1** | flexion100-h15 | ManiFlow 레포 run, `run_system.py`가 자동 탐색 | 100 fps 예측기, denoise 1. H0 = flexion40(50 ms 격자, 폐기) |
| L | **L1** | llp_v13 = 진자 v4 @ epoch 6,000 | `checkpoints/L1_llp_pendulum/` | 랜덤화 진자 학습, 40 N·m, 10 ms 제어. L2~L5는 실패(results/에만) |
| W | **W1** | tracker v33 | `checkpoints/W1_wearer_human/` | 사람 범위 근력(hip 한계 200, 주기 피크 0.95 N·m/kg). W0 = v32(기준선용) |

**학습 플랜트 = 랜덤화 진자, 스켈레톤 = 시험장.** L1은 스켈레톤·mocap·걷기·달리기를 한 번도 보지 않았다. 진자(고정 골반 hip 1축)에
관성 ×1~25, 착용자 PD ω_n 12~24, 입각 사건(외부 하중 ≤120 N·m + 굳힘), 작은 섭동, 다양한 의도 궤적(sine + 고조파 + hold)을
섞고, HLP는 배포 형식(10 ms × 6점)으로 에뮬레이션했다. 비용 = 추종 + |τ·ω| + τ² + 보조 토크 절약 + action 변화.
세부: `checkpoints/L1_llp_pendulum/INFO.md`, `examples/experiments/assist_pendulum/mlp_v4.py` docstring.

---

## 4. 어떻게 여기까지 왔나 (타임라인)

| 날짜 | 일 | 결과 |
|---|---|---|
| ~09-08 | 진자 LLP v3 + 실 HLP 결합(Phase C) | 진자 안에서는 동작. 스켈레톤에 옮기면 방향이 안 맞음(상관 0.19) |
| 09-12 | 통합 시스템 구축: 독립 제어기 `protomotions/assist/`, 서브스텝 훅, 스켈레톤 A/B 러너 | 폐루프 동작. 착용자 tracker 재학습(v30 walk, v31 15모션) |
| 09-13~14 | **개인화** 시도: 스켈레톤 위에서 LLP 추가 학습 llp_v3~v6, 두 속도 구조(HLP 10 ms, LLP 5 ms) v10, 대칭 보행 착용자 W0(v32) | \|τ·ω\| 기준 −22~−32 %로 보였음 |
| 09-15 | **오라클 진단** | \|τ·ω\|는 각속도만 줄어도 내려감. τ²로 재면 v10은 +7 %. **목적함수가 틀렸었다.** β 오라클 상한 τ² −32 % |
| 09-16 | v11(τ² 항 + 한계 해제), v12(80 N·m) | 걷기는 되지만 **달리기는 아무도 못 도움**(v11 2 s 낙상, v12 +9 %) |
| 09-17~22 | 진자 단계 재점검, 착용자 토크 스파이크 분석 | 스파이크는 사람이 아니라 tracker + 강성 PD의 산물. 진자가 "밀기만 하는 그네"를 가르쳤음 |
| 09-22 | **진자 v4 = 랜덤화 플랜트** 설계·학습 시작 | 하중·관성·의도 다양화, 배포 HLP 형식 일치 |
| 09-23 | **전이 확인** | 스켈레톤을 본 적 없는 v4 epoch 6,000이 W0에서 걷기 τ² −30.5 / 달리기 −25.5 %. 달리기를 도운 첫 정책. 10k·14k는 물러남 |
| 09-28 | v4 epoch 6,000 보관(L1). v5(토크 절약 항 축소) | v5도 후기 하락 → 절약 항이 원인 아님. 중단 |
| 09-28 | **사람 범위 착용자 W1** 학습·보관(hip 한계 200) | W1에서 L1 그대로는 걷기 무효·달리기 낙상, **×0.5면 −19.3 / −23.9 %**. β −35 / −41 |
| 09-29 | 버전 체계 H/L/W, 용어 통합·개인화 정리 | |
| 09-29 밤 | 진자 변형 **L2~L5** 동시 학습·W1 평가, 유각 말기 마스크 진단 | 전부 L1×0.5 아래. 교훈 §5. 전부 중단 |
| 09-30 | (가)/(나) 선택지 설명, "현재 최선 = L1×0.5" 확인. HLP 축소 실험은 불필요로 취소 | 결정 대기 |

---

## 5. 확정된 사실·교훈 (다시 배우지 말 것)

1. **잣대는 착용자 hip 토크의 τ²**(근활성 프록시). \|τ·ω\|(파워)는 착용자 다리가 덜 흔들리기만 해도 내려가 속는다.
   항상 τ², 전진 거리, 몸체 추종 오차, 낙상을 **한 표에서** 본다(`compare_runs.py`). 재실행 산포는 1~2 %.
2. **비교는 A(무보조) vs B(보조)만.** mocap과 비교하지 않는다. 변화는 "두 값 그대로"나 절대 단위로 쓰고 "pt"는 쓰지 않는다(사용자 지시).
3. **정책은 착용자 근력을 못 본다.** 각도만 보므로 보조 크기는 착용자별 배율로 정한다. W1에서 0.5. 사람 범위에 40 N·m는 과보조.
4. **진자 정책의 쓸모는 "하중에 맞서 의도의 토크를 착용자 대신 내는 것"에서 온다.** 하중을 줄이거나(L3), 환경이 대신 내거나(L5),
   chunk에서 하중 신호를 감추면(L4) 학습 신호가 사라진다. **진자를 "쉽게" 만드는 방향은 다시 시도하지 말 것.**
5. **bang-bang(포화)은 필요량 > 한계의 결과**다. 한계를 랜덤화해도(L2) 안 바뀐다.
6. **유각 말기 브레이크**: L1은 의도 반전 전에 관성을 되돌리는 토크를 미리 낸다. 걷기에선 착지 직전(90~100 %) 도움, 달리기에선
   70~100 % 전체가 손해(막으면 −24 → −30). β와의 나머지 차이는 착지 구간 비례성(β 이득 −486 vs L1 −211, RMS는 비슷).
7. **β 오라클·마스크 진단은 특권 정보**(착용자 토크, 보행 위상)를 쓴다. 상한과 인과를 보는 도구지 배포 후보가 아니다.
8. **HLP 형식은 학습·배포가 같아야 한다.** 옛 형식(25 ms × 7점)으로 학습한 v4는 전이가 안 됐고, 배포 형식(10 ms × 6점)으로 맞춘 뒤 됐다.
9. **평가는 학습 제어 주기와 `--llp-dt`를 맞춘다**(L1 = 0.01). 영점은 `--calibration wearer`(착용자 스트림에서 측정).
10. **40 N·m는 스펙이 아니라 우리가 고른 값**(25 → 40). 실제 구동기의 피크·연속 토크·대역폭을 받으면 한계와 `--torque-lpf-hz`를 그 값으로.
    측정: 12 Hz 저역통과는 거의 무손실, 6 Hz −4~5, 3 Hz 붕괴.
11. **W0 수치는 비교 불가.** 사람보다 센 착용자가 과보조·오방향을 받아 준 결과. 모든 LLP 평가는 W1에서.
12. **Newton은 실행 간 bit-reproducible이 아니다.** 회귀 판정은 재실행 산포로.
13. 진자 v4 전용 세그폴트(매 step 게인 통지가 MuJoCo 상수 재계산)는 408f0b6으로 고쳤다. 스켈레톤 run 세그폴트는 원인 미확정(CUDA graph 1순위).
14. HLP H1(6블록 × 256, 10.5 M)은 이미 작다. 크기 축소 실험은 불필요로 취소(2026-09-30). ARDY는 HLP 후보에서 제외(2026-09-08).

---

## 6. 평가·학습 방법

모두 conda env `sbc` (`~/miniconda3/envs/sbc/bin/python`). ManiFlow **학습**만 env `maniflow`.
추론·평가에서는 `--robot-name`을 주지 않는다(로봇 config는 체크포인트의 `resolved_configs_inference.pt`에서 읽음).
음수 CLI 값은 `--flag=-5e-6` 형태로.

```bash
# 통합 평가 (= 스켈레톤 A/B). motion-id 0 = 걷기, 1 = 달리기. 결과는 results/oracle/<날짜>_tracker_<태그>_m<모션>/
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/W1_wearer_human/last.ckpt \
    --llp-checkpoint checkpoints/L1_llp_pendulum/last.ckpt --llp-dt 0.01 --assist-scale 0.5 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40 \
    --calibration wearer --out results/oracle --tag L1s05        # --record = mp4, --torque-lpf-hz 12 = 구동기 대역
# 여러 실행을 한 표에 (τ², |τ·ω|, 속도, 몸체 오차, 보조 RMS, 상관)
python examples/experiments/assist_skeleton/compare_runs.py results/oracle/*_tracker_v33e12218_*_m0
# 착용자 자체가 얼마나 사람 같은가 (스윙, 토크 피크 N·m/kg, 피크/RMS)
python examples/experiments/assist_skeleton/eval_wearer.py results/oracle/<run>
# 보고서: 여러 실행 비교 HTML / 체크포인트 하나 깊게 (위상표·3보 시계열·학습 곡선)
python examples/experiments/assist_skeleton/report_html.py --out docs/transfer_report.html --walk <runs...> --run <runs...>
python examples/experiments/assist_skeleton/report_checkpoint_html.py --out docs/llp_v13_report.html --walk <run> --run <run>
# 특권 탐침: β 오라클(상한), 위상 창 토크 강제, 정책 토크 위상 마스크
python examples/experiments/assist_skeleton/run_system.py ... --oracle beta --oracle-beta 0.5
python examples/experiments/assist_skeleton/run_system.py ... --mask-window 70,100
# 학습 중 체크포인트가 생기면 W1에서 자동 평가 + 비교표 (LIMITS, TRACKER, CALIB 환경변수)
bash examples/experiments/assist_skeleton/eval_when_ready.sh results/llp_L6_pendulum/epoch_2000.ckpt L6e2k aisbc-llp-L6
```

```bash
# LLP 학습 (랜덤화 진자). 장시간 학습은 반드시 systemd 사용자 서비스로 (세그폴트 자동 재시작)
systemd-run --user --unit=aisbc-llp-L6 --working-directory="$PWD" \
  --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
  --property=StandardOutput=append:$PWD/results/_pids/llp_L6.log \
  --property=StandardError=append:$PWD/results/_pids/llp_L6.log --setenv=PYTHONFAULTHANDLER=1 \
  ~/miniconda3/envs/sbc/bin/python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
    --experiment-path examples/experiments/assist_pendulum/mlp_v4.py --experiment-name llp_L6_pendulum \
    --motion-file none --num-envs 4096 --batch-size 16384      # + mlp_v4.py의 CLI 옵션 (기본값 = L1)
systemctl --user status aisbc-llp-L6 ; systemctl --user stop aisbc-llp-L6
# 가상 착용자 학습 (W1 레시피, W0에서 warm start)
bash examples/experiments/assist_skeleton/train_tracker_human.sh
# 학습 전 진자 env 점검 그림
python examples/experiments/assist_pendulum/check_env_v4.py
```

L1 학습 속도: 4096 env에서 epoch당 약 6 s, 6,000 epoch ≈ 10 h. 평가는 한 실행 40 s ≈ 3분. 전이 성능은 epoch 6,000 근방이
정점이고 그 뒤 물러나므로 **2k / 4k / 6k에서 W1 평가**를 보고 판단한다.

---

## 7. 열린 과제와 다음 선택지

**이전 제안 (09-30까지; 10-01에 일반화 개선 진행으로 결정)**

- **(가) H1 + L1 × 0.5 @ W1을 통합 결과로 확정.** 학습 없이 평가·문서만: 배율 0.3~0.7 탐색(실행당 3분), `--torque-lpf-hz 12`에서도
  유지되는지, W1 기준 보고서(지금 보고서는 W0 기준 + W1 부록 한 줄), 버전 문서에 "배포 후보 = H1+L1×배율, W1에서 0.5" 명시.
  남는 한계: β 상한의 절반 남짓, 배율은 사람마다 손으로 맞춰야 함(목표의 "크기" 부분 양보), 달리기 유각 말기 오방향은 그대로.
- **(나) 달리기 유각 말기 시점만 겨냥한 진자 변형(L6).** 가설: "반전이 느리면(걷기) 브레이크, 빠르면(달리기) 브레이크 없음"을
  정책이 chunk 모양에서 구분하게 한다. L5의 "환경이 반전 관성을 대신 냄"을 **의도 반전 가속도가 문턱 이상일 때만** 켠다. 문턱은
  W1 걷기·달리기 기록에서 chunk 반전 가속도를 재어 두 분포 사이에 둔다. **학습 전에** 같은 규칙을 L1 위 추론 시 규칙으로 얹어
  (정책 관측만 사용) W1에서 확인하고, 달리기가 −30 쪽으로 가고 걷기가 유지될 때만 학습을 건다. 상한은 마스크 진단의 달리기 −30 %,
  걷기는 현상 유지. 위험: L5처럼 학습 신호 소실, 걷기·달리기 반전 속도 분포 겹침.
- 순서 제안: (가)의 배율 탐색·대역 확인은 학습이 없으니 어느 쪽이든 먼저. 그 뒤 (나)의 학습 전 시험.

**그 외 (우선순위 낮음)**
- `protomotions/assist/verify_controller.py`: **10-01 L1/H1 기준 PASS 완료**(위 추가 기록).
  스켈레톤 env ↔ `AssistController` 전체 등가성 검사는 별도 남음.
- 구동기 스펙(피크·연속 토크·대역폭) 반영. 달리기는 40 N·m에서 한계 도달 64 %라 한계가 병목.
- HLP 학습셋의 계단 부족(74,488클립 중 98개) → 계단·전환에서 오차 2.3~2.6배.
- W1의 피크/RMS 3.9: 200 N·m 안에서도 착지 슬램이 남아 있음(τ² 벌점 −1e-6으로는 못 눌렀다).
- 스켈레톤 run 세그폴트 원인.

---

## 8. 파일 지도 (요약 — 전체는 FRAMEWORK §4)

| 경로 | 내용 |
|---|---|
| `checkpoints/L1_llp_pendulum/`, `W1_wearer_human/` | 현재 보관본 + INFO.md. 나머지 `llp_v3~v10`, `v18~v32`는 legacy |
| `results/` (git 미추적) | 학습 run과 평가 기록. `results/oracle/` = 통합 평가 104회(`<날짜>_tracker_<착용자>_<태그>_m<모션>/` 안에 traces·metrics·report). `results/_pids/*.log` = systemd 로그. `llp_L2~L5_pendulum/`, `assist_pendulum_v4/v5`, `mimic_suit_human/`(W1 run) |
| `examples/experiments/assist_pendulum/` | 진자 LLP 학습: `mlp_v4.py`(L1 + L2~L5 옵션, docstring에 run 기록), `check_env_v4.py`, `eval_abc.py` |
| `examples/experiments/assist_skeleton/` | 통합 평가·착용자: `run_system.py`(A/B 러너, 오라클·마스크·배율·LPF 옵션), `compare_runs.py`, `report_html.py`, `report_checkpoint_html.py`, `eval_wearer.py`, `eval_when_ready.sh`, `tracker_human.py` + `train_tracker_human.sh`(W1), `llp_walk.py`(개인화, 미사용), `make_gait_motions.py`, `README.md` |
| `protomotions/assist/` | 독립 제어기(`controller.py`), LLP actor 로더, 스켈레톤 특징량·영점, 등가성 검증 |
| `protomotions/envs/base_env/assist_env.py`, `protomotions/envs/control/assist_*_control.py` | 진자 env(관성·질량·한계 랜덤화, 보조 주입), 외란(입각 하중·굳힘·반전 브레이크), 의도·chunk 생성 |
| `protomotions/envs/base_env/skeleton_assist_env.py` | 개인화 env(미사용) |
| `protomotions/simulator/newton/substep_hook.py` | env step 안 10 ms 콜백(LLP) |
| `protomotions/robot_configs/skeleton_torque_suit_human.py`, `hip_pendulum*.py` | W1 로봇 config(hip 한계 200), 진자 |
| `protomotions/data/assets/mjcf/skeleton_torque_suit.xml` | 인체 모델(뼈 메시 그리기 전용 + 캡슐 충돌; headless는 메시 안 읽음) |
| `data/motion_for_trackers/skeleton_torque_suit_gait.pt` | 보행 세트(walk 0.75 / run 0.25). motion 0 = 걷기, 1 = 달리기 |
| `protomotions/maniflow/` | HLP 접근 계층(README 참고). `import maniflow` 직접 금지 |
| `docs/llp_v13_report.html`, `docs/transfer_report.html` | L1 깊은 보고서(W0 기준 + W1 부록), 여러 체크포인트 비교. 같은 내용의 claude.ai artifact: `A1enWCcYbiRaEE6rFNn6qB`, `LhmuUcPkK1iZxuKRmCvoNp` |

Claude Code 에이전트 메모리(이 레포 전용): `~/.claude/projects/-home-bak-Projects-AI-SBC-PROTOMOTIONS/memory/` — 교훈·사용자 지시가
파일 하나당 하나씩. 다른 에이전트도 읽어도 된다. 핵심은 이 문서 §5·§9에 옮겨 두었다.

---

## 9. 운영 규칙과 사용자 방침

- **장시간 학습은 systemd 사용자 서비스**(§6 명령)로. `nohup`/`setsid`는 세그폴트에 죽은 채 방치된다. `pkill -f <이름>`은 셸까지 죽인다.
- 학습 중 산출물은 `results/<exp>/`, 보관은 `checkpoints/<버전>/ + INFO.md`(가중치는 LFS). 보관하지 않은 run은 지워도 된다.
- 커밋: `git commit -s`, gitmoji + 영어 제목, 본문 끝 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. 1인 레포라 main에 직접.
- `pre-commit run --files <파일>`만. `--all-files` 금지. Apache 헤더는 기존 파일에서 그대로 복사.
- **사용자와는 한국어.** 설명은 은어·단계 라벨 없이 내용으로("스모크", "Phase D" 같은 말 금지), 기술 용어는 통용 영어 그대로.
  코드 설명은 계층 분류 대신 실행 순서의 file:line 트레이스로.
- 사용자 성향: 하드웨어 한계를 고정하기보다 학습에서 **랜덤화**하는 쪽을 선호. 필터는 자발적으로 걸지 않는다(구동기 스펙이 오면 그 값으로).
  먼저 제안하지 말 것: ARDY, HLP 크기 축소, 진자를 쉽게 만드는 변형.
- 사용자는 원격 tmux로도 작업한다. 남의 세션·유닛을 끌 때는 먼저 확인.

---

## 10. 보고서·기록 위치

| 무엇 | 어디 |
|---|---|
| 실험 일지(가장 상세) | `docs/NEXT_SESSION.md` §2~§2-L (오라클 → v11/v12 → 진자 v4 → v5 → W1 → L2~L5), §3 한계 출처, §4 후보, §5 운영, §6 방침 |
| 버전·용어 | `docs/AI_SBC_VERSIONS.md` §0~§5, `docs/versions.html` |
| 설계·함정·코드 읽기 | `docs/AI_SBC_FRAMEWORK.md` |
| 보관본 설정·결과 | `checkpoints/L1_llp_pendulum/INFO.md`, `checkpoints/W1_wearer_human/INFO.md` |
| 커밋 메시지 | `git log --date=short` — 2026-09-13 이후 각 실험의 결론이 제목에 있다 |
| 그림 | `docs/*.png` (hlp_llp_timing, torque_phase, pendulum_vs_walking_torque, oracle, landing_torque_zoom) |
