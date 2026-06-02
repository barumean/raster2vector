# raster2vector

흑백 래스터 도면(PNG, JPG, BMP 등)을 DXF 벡터 파일로 변환하는 CLI 도구입니다.  
건축 단면도, 평면도, 기계 도면 등 스캔 도면에 최적화되어 있습니다.

---

## 설치 (Installation)

### 요구사항 (Requirements)
- Python 3.10 이상
- 의존 패키지: `requirements.txt` 참조

### 설치 절차

```bash
# 1. 저장소 클론
git clone https://github.com/barumean/raster2vector.git
cd raster2vector

# 2. 가상환경 생성 및 활성화 (권장)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt
```

### Windows 유의사항
`scikit-image`가 설치되면 스켈레톤화 품질이 향상됩니다.  
설치 실패 시에도 내장 Zhang-Suen 알고리즘으로 동작합니다.

---

## 기본 사용법 (Quick Start)

```bash
# 가장 기본적인 변환
python raster2vector.py drawing.jpg

# 단면도 권장 설정 (텍스트 분리 + arc 반경 필터 + 중복선 제거)
python raster2vector.py section.jpg --text-separation --min-arc-radius 15 --verbose

# 스캔 도면 (DPI 명시, 미리보기 생성)
python raster2vector.py scanned.jpg --dpi 300 --preview --verbose

# 결과 파일명 지정
python raster2vector.py drawing.png -o output.dxf
```

---

## 출력 레이어 구조 (Output Layers)

| 레이어 | 색상 | 내용 |
|--------|------|------|
| `LINES` | 빨강 | 직선 세그먼트 (LINE 엔티티) |
| `CONTOURS` | 초록 | 곡선/복합 폴리라인 (LWPOLYLINE) |
| `ARCS` | 파랑 | 검출된 원호/원 (ARC, CIRCLE) |
| `BOXES` | 청록 | 닫힌 직사각형 윤곽 |
| `BOX` | 흰/검정 | 도면 전체 외곽 경계 |
| `DASHED` | 빨강 | 파선/숨은선 |
| `TEXT_CANDIDATES` | 노랑 | 텍스트 영역 후보 (`--text-separation` 사용 시) |
| `ELONGATED` | 마젠타 | 세장형 획 후보 |

---

## 전체 옵션 (All Options)

### 전처리 (Preprocessing)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--threshold-method {otsu,adaptive,sauvola}` | `otsu` | 이진화 방법. 조명 불균일한 스캔에는 `sauvola` 권장 |
| `--threshold 0-255` | — | 수동 임계값 (설정 시 `--threshold-method` 무시) |
| `--invert` | off | 반전 이미지 (흰 배경에 흰 선인 경우) |
| `--morph {none,open,close}` | `none` | 형태학적 연산 (얇은 선 손상 주의) |
| `--morph-kernel N` | `2` | 형태학 커널 크기 (px) |
| `--no-despeckle` | off | 미세 잡음 제거 비활성화 |
| `--min-speckle-area N` | `3` | 이 면적 미만의 고립 점 제거 (px²) |
| `--deskew` | off | 수평 기울기 자동 보정 |
| `--blur-radius N` | `0` | 에지 보존 가우시안 블러 반경 (0=비활성, 권장 1-3) |
| `--blur-delta N` | `20` | 블러 적용 임계 강도차 (imagetracerjs blurdelta) |
| `--adaptive-block-size N` | `51` | 적응형 이진화 블록 크기 (홀수) |
| `--adaptive-c N` | `9` | 적응형 이진화 보정값 |
| `--sauvola-window N` | `25` | Sauvola 로컬 윈도우 크기 |
| `--sauvola-k F` | `0.2` | Sauvola k 파라미터 (낮을수록 전경 증가) |

### 벡터화 (Vectorisation)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--min-contour-length F` | `15.0` | 최소 컨투어 호 길이 (px, 미만 제거) |
| `--min-contour-area F` | `10.0` | 최소 컨투어 바운딩박스 면적 (px²) |
| `--max-line-deviation F` | `2.0` | 컨투어를 직선으로 분류하는 최대 수직 편차 (px) |
| `--no-hough` | off | 보조 Hough 직선 검출 비활성화 |
| `--min-line-length PX` | `80` | Hough 최소 선분 길이 |
| `--max-gap PX` | `15` | Hough 선분 내 최대 간격 |
| `--hough-threshold N` | `30` | HoughLinesP 누산기 임계값 |
| `--canny-low N` | `50` | Canny 하위 임계값 |
| `--canny-high N` | `150` | Canny 상위 임계값 |
| `--approx-epsilon F` | auto | Douglas-Peucker 단순화 허용오차 (기본 이미지 대각선의 0.3%) |
| `--pre-close-kernel N` | `0` | 에지 검출 전 형태학적 닫힘 (두꺼운 선 도면에 5-15 권장) |
| `--snap-radius F` | `4.0` | 근접 끝점 스냅 거리 (px, 0=비활성) |
| `--no-merge-lines` | off | Hough 공선 세그먼트 병합 비활성화 |
| `--no-dedup-lines` | off | 중복 평행선 제거 비활성화 (두꺼운 선의 이중선 남김) |

### 원호/원 검출

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--no-arcs` | off | 원호/원 검출 비활성화 (모두 폴리라인으로 출력) |
| `--arc-tol F` | auto | 원 피팅 최대 RMS 잔차 (기본 이미지 대각선의 0.5%) |
| `--min-arc-radius PX` | `0.0` | 최소 원호 반경 (px). **텍스트 문자 arc 억제에 효과적** (예: `--min-arc-radius 15`) |
| `--corner-threshold DEG` | `60.0` | 분할 arc 추출 코너 검출 임계각도 |
| `--splice-threshold DEG` | `45.0` | arc 분할 최대 각도 범위 |

### 박스/사각형 검출

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--no-detect-boxes` | off | 직사각형 윤곽 분류 비활성화 |
| `--box-angle-tol DEG` | `20.0` | 사각형 인식 각도 허용오차 |

### 세그먼트 통합

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--no-consolidate` | off | 교차-컨투어 세그먼트 통합 비활성화 |
| `--consolidate-perp-tol PX` | `6.0` | 통합 수직 거리 허용오차 |
| `--consolidate-angle-tol DEG` | `4.0` | 통합 각도 허용오차 |

### 구조 정리 (Structure Cleanup)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--structure-cleanup` | off | 직선에 가까운 폴리라인을 직선으로 단순화 (대각선 각도 유지) |
| `--structure-line-tolerance F` | `2.5` | 단순화 허용 편차 (px) |
| `--no-quad-detection` | off | 사각형 폐곡선 정리 비활성화 |

### 후처리 (Post-processing)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--gap-jump` | off | 근접 끝점 간격 자동 연결 (Scan2CAD gap_jump) |
| `--gap-px PX` | `15.0` | 연결할 최대 간격 (px) |
| `--fan-angle DEG` | `20.0` | 방향 탐색 반각도 |
| `--orthogonalize` | off | 수평/수직에 가까운 선 직교 스냅 |
| `--ortho-accuracy DEG` | `2.0` | 직교 스냅 각도 허용오차 |
| `--ortho-base-angle DEG` | `0.0` | 기준축 각도 |
| `--detect-dashes` | off | 파선/숨은선 감지 및 DASHED 레이어 분리 |
| `--max-dash-len PX` | `40.0` | 파선 후보 최대 선분 길이 |
| `--right-angle-enhance` | off | 근사 직각 코너를 정확한 직각으로 스냅 |
| `--right-angle-tol DEG` | `10.0` | 직각 스냅 허용오차 |
| `--remove-staircase` | off | 1px 대각선 계단 아티팩트 제거 (저DPI 스캔) |
| `--text-separation` | off | 텍스트 영역을 TEXT_CANDIDATES 레이어로 분리 |
| `--stroke-width` | off | 선폭 추정 및 DXF 선가중치 설정 |
| `--no-page-border` | off | 전체 도면 외곽 경계(BOX 레이어) 생성 비활성화 |

### 출력

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dpi F` | `96.0` | 소스 이미지 해상도 (px→mm 변환에 사용) |
| `--preview` | off | 검출 결과 미리보기 PNG 저장 |
| `--output-preview PATH` | `<stem>_preview.png` | 미리보기 경로 |
| `--verbose` | off | 처리 통계 출력 |

---

## 사용 예제 (Usage Examples)

### 단면도 (Cross-section drawings) — 권장 설정
```bash
# 텍스트 분리 + 소형 arc 억제 + 중복선 제거
python raster2vector.py section.jpg \
  --text-separation \
  --min-arc-radius 15 \
  --dpi 150 \
  --verbose

# 파선(숨은선) 포함 단면도
python raster2vector.py section.jpg \
  --text-separation \
  --min-arc-radius 15 \
  --detect-dashes \
  --dpi 150
```

### 스캔 도면 품질 개선
```bash
# 불균일 조명 스캔 (Sauvola 이진화)
python raster2vector.py scanned.jpg \
  --threshold-method sauvola \
  --deskew \
  --blur-radius 2 \
  --dpi 300

# 두꺼운 선 도면 (이중선 제거)
python raster2vector.py thick.jpg \
  --pre-close-kernel 7 \
  --no-dedup-lines
```

### 문제 해결 팁

| 증상 | 권장 옵션 |
|------|-----------|
| 텍스트 문자가 arc으로 검출됨 | `--min-arc-radius 15` 또는 `--text-separation` |
| 박스 경계선이 이중/삼중으로 나옴 | `--no-dedup-lines` 제거 (기본 활성), `--pre-close-kernel 5` |
| 선이 너무 많이 끊김 | `--gap-jump --gap-px 20` |
| 너무 많은 잡선 검출 | `--min-contour-length 30 --min-contour-area 50` |
| 결과가 텅 빔 | `--invert` 또는 `--threshold-method adaptive` |
| 사각형이 CONTOURS에 섞임 | `--detect-boxes` (기본 활성 확인) |

---

## 아키텍처 (Architecture)

```
raster2vector.py          CLI 진입점 (argparse, 파이프라인 오케스트레이션)
src/
  preprocessor.py         이미지 로드, 이진화, 형태학적 정리, 기울기 보정
  vectorizer.py           Canny 에지, 컨투어 추출, Hough 보조, arc 피팅,
                          직사각형 분류, 세그먼트 통합, 구조 정리
  dxf_exporter.py         ezdxf 문서 생성, 레이어 설정, DXF 내보내기
  text_separator.py       텍스트/그래픽 분리, 연결 컴포넌트 분석
  stroke_width.py         중심축 기반 선폭 추정 (SPV)
tests/
  test_smoke.py           합성 이미지 스모크 테스트 (73개)
```

### 처리 파이프라인

```
입력 이미지
  → 전처리 (이진화, 기울기 보정, 잡음 제거)
  → [선택] 텍스트 영역 분리
  → Canny 에지 검출 + 1px 세선화
  → findContours (주 지오메트리)
  → Douglas-Peucker 단순화
  → 직선 / 폴리라인 / arc 분류
  → [보조] HoughLinesP (긴 직선 보완)
  → 중복선 제거, 끝점 스냅
  → [선택] gap-jump, 직교화, 파선 검출
  → [선택] 세그먼트 통합 (cross-contour)
  → [선택] 직사각형 분류 → BOXES 레이어
  → DXF 내보내기 (레이어별 분리)
```

---

## 테스트 실행

```bash
pytest tests/test_smoke.py -v
```

---

## 라이선스

MIT License
