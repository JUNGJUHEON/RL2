import subprocess
import socket
import time
import cv2
import os
import sys
import shutil
import psutil
import atexit
import numpy as np

from src.envs.env import ParallelEnvironment
from src.envs.ab.agent_client import AgentClient, GameState

COMMUNICATION_ERRORS = (TimeoutError, OSError, ConnectionResetError, BrokenPipeError)

# State space
STATE_PIXEL_RES = 128  # width and height of (preprocessed) states

# Bird type one-hot encoding
# 레벨 XML에서 읽은 새 순서를 5-dim one-hot으로 인코딩
BIRD_TYPES    = ['BirdRed', 'BirdBlue', 'BirdYellow', 'BirdBlack', 'BirdWhite']
BIRD_TYPE_IDX = {b: i for i, b in enumerate(BIRD_TYPES)}
BIRD_DIM      = len(BIRD_TYPES)   # 5

# Action space
ANGLE_RESOLUTION = 20  # the number of possible (discretized) shot angles
TAP_TIME_RESOLUTION = 10  # the number of possible tap times
MAXIMUM_TAP_TIME = 4000  # maximum tap time (in ms)
PHI = 10  # dead shot angle bottom (in degrees)
PSI = 40  # dead shot angle top (in degrees)
ACTIONS = []

SERVER_CLIENT_CONFIG = {
    "requestbufbytes": 4,
    "d": 4,
    "e": 5
}

# Reward
SCORE_NORMALIZATION = 10000
WIN_BONUS = 5.0
LOSS_PENALTY = 1.0
SHOT_PENALTY = 0.05
USE_REWARD_CLIP = False
REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 10.0

TOTAL_LEVEL_NUMBER = 1300  # non-novelty levels
LIST_OF_VALIDATION_LEVELS = []

def _linux_ab_dir() -> str:
    """Linux 빌드가 들어있는 디렉터리의 절대 경로."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ab", "Science Birds 0.3.8", "linux",
    )


def _generate_linux_config(linux_dir: str) -> None:
    """Windows 레벨 3종(type00/type01/type13) 총 400개를 단일 trial에 담은 config.xml을 생성.

    Linux Unity(9001.x86_64)의 GameLevelSetInfo.getLevelSetXmlData()는 파일명에서
    언더스코어 위치로 Substring을 계산하므로, 'level-00.xml' 형식은 크래시를 일으킨다.
    해결: type2_win/ 아래에 '00001_1_1_2_0.xml' 형식으로 파일을 실제 복사한다.
    심볼릭 링크는 Unity가 따라가지 못할 수 있으므로 사용하지 않는다.

    인덱스 매핑 (Windows와 동일):
      00001–00200 → type00-givenLevel        (게임 인덱스 1-200)
      00201–00300 → type01-1Pig-0TNT         (게임 인덱스 201-300)
      00301–00400 → type13-1Pig-1TNT-standard (게임 인덱스 301-400)
    """
    win_base = os.path.join(
        os.path.dirname(linux_dir),
        "Science Birds_Data", "StreamingAssets", "Levels", "novelty_level_0",
    )
    sources = [
        ("type00-givenLevel",                1),
        ("type01-1Pig-0TNT-standardBlocks", 201),
        ("type13-1Pig-1TNT-standard",       301),
    ]

    type2_win_levels = os.path.join(
        linux_dir, "Levels", "novelty_level_0", "type2_win", "Levels"
    )
    os.makedirs(type2_win_levels, exist_ok=True)

    all_entries = []   # list of (global_idx, src_path)
    for type_dir, start_idx in sources:
        level_dir = os.path.join(win_base, type_dir, "Levels")
        if not os.path.isdir(level_dir):
            print(f"  [경고] 레벨 디렉터리를 찾을 수 없음: {level_dir}")
            continue
        xml_files = sorted(
            (f for f in os.listdir(level_dir) if f.endswith(".xml")),
            key=lambda x: int(x.replace("level-", "").replace(".xml", "")),
        )
        for n, fname in enumerate(xml_files):
            all_entries.append((start_idx + n, os.path.join(level_dir, fname)))

    copied = 0
    for global_idx, src_path in all_entries:
        type2_name = f"{global_idx:05d}_1_1_2_0.xml"
        dst_path = os.path.join(type2_win_levels, type2_name)
        if os.path.islink(dst_path) or not os.path.isfile(dst_path):
            if os.path.islink(dst_path):
                os.remove(dst_path)
            shutil.copy2(src_path, dst_path)
            copied += 1
    if copied:
        print(f"  type2_win/ 레벨 파일 복사: {copied}개")

    lines = [
        '<?xml version="1.0" encoding="utf-16"?>',
        '<evaluation>',
        '  <novelty_detection_measurement step="1" measure_in_training="True" measure_in_testing="True" />',
        '  <trials>',
        '    <trial id="0" number_of_executions="1" checkpoint_time_limit="10000"'
        ' checkpoint_interaction_limit="10000" notify_novelty="False">',
        '      <game_level_set mode="training" time_limit="999999"'
        ' total_interaction_limit="999999" attempt_limit_per_level="999"'
        ' allow_level_selection="True">',
    ]
    for global_idx, _ in all_entries:
        type2_name = f"{global_idx:05d}_1_1_2_0.xml"
        lines.append(
            f'        <game_levels level_path="./Levels/novelty_level_0/type2_win/Levels/{type2_name}" />'
        )
    lines += [
        '      </game_level_set>',
        '    </trial>',
        '  </trials>',
        '</evaluation>',
    ]

    config_path = os.path.join(linux_dir, "config.xml")
    content = "\n".join(lines) + "\n"
    with open(config_path, "w", encoding="utf-16") as f:
        f.write(content)
    print(f"  Linux config.xml 생성됨: type2_win/ → {len(all_entries)}개 레벨 (type00/type01/type13)")


def _parse_birds_from_xml(xml_path: str, re_module) -> list:
    """XML 파일에서 <Birds> 섹션의 새 타입 목록을 파싱해 반환."""
    with open(xml_path, 'rb') as f:
        raw = f.read()
    text = raw.replace(b'\x00', b'').decode('ascii', errors='ignore')
    m = re_module.search(r'<Birds>(.*?)</Birds>', text, re_module.DOTALL)
    if m:
        return re_module.findall(r'<Bird type="(\w+)"', m.group(0))
    return []



def _build_windows_level_bird_map() -> dict:
    """Windows 빌드용: 레벨 번호 → 새 발사 순서 리스트를 XML에서 읽어 반환."""
    import re as _re
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ab", "Science Birds 0.3.8",
        "Science Birds_Data", "StreamingAssets", "Levels",
        "novelty_level_0",
    )
    sources = [
        ("type00-givenLevel",               1, 200),
        ("type01-1Pig-0TNT-standardBlocks", 201, 300),
        ("type13-1Pig-1TNT-standard",       301, 400),
    ]
    bird_map = {}
    for type_dir, start_idx, end_idx in sources:
        level_dir = os.path.join(base, type_dir, "Levels")
        for n in range(end_idx - start_idx + 1):
            xml_path = os.path.join(level_dir, f"level-{n:02d}.xml")
            if not os.path.isfile(xml_path):
                continue
            birds = _parse_birds_from_xml(xml_path, __import__('re'))
            if birds:
                bird_map[start_idx + n] = birds
    return bird_map


# ── 레벨 데이터 초기화 ─────────────────────────────────────────────────────
# 두 플랫폼 모두 Windows type00-givenLevel XML에서 새 종류를 읽음
LEVEL_BIRD_MAP: dict = _build_windows_level_bird_map()

# ── 필터링된 훈련/평가 레벨 풀 (총 200개) ────────────────────────────────
# type00-givenLevel (1-200) 중 난이도 분석으로 제외된 67개 위치 (1-indexed)
_GIVEN_LEVEL_EXCLUDED = frozenset([
     3,  8, 11, 23, 25, 32, 38, 44, 45, 65,
    80, 81, 82, 87, 89, 93, 94, 96,101,106,
   109,123,127,130,131,133,134,136,137,139,
   141,146,150,151,152,153,154,155,156,158,
   160,164,165,166,169,170,171,172,173,174,
   175,177,178,179,180,182,183,186,188,189,
   192,194,195,196,197,198,199,
])
_n_excluded = len(_GIVEN_LEVEL_EXCLUDED)          # 67
_n_type01   = (_n_excluded + 1) // 2              # 34  (type01-1Pig-0TNT)
_n_type13   = _n_excluded       // 2              # 33  (type13-1Pig-1TNT)

# Linux와 Windows 동일한 200개 필터링 레벨 풀 사용
# type00-givenLevel: 게임 인덱스 1-200  (133개 after exclusion)
# type01:            게임 인덱스 201-300 (34개 사용)
# type13:            게임 인덱스 301-400 (33개 사용)
FILTERED_TRAIN_LEVELS: list = sorted(
    [i for i in range(1, 201) if i not in _GIVEN_LEVEL_EXCLUDED]
    + list(range(201, 201 + _n_type01))
    + list(range(301, 301 + _n_type13))
)


def angle_to_vector(alpha):
    rad_shot_angle = np.deg2rad(alpha)

    dx = - np.sin(rad_shot_angle) * 80
    dy = np.cos(rad_shot_angle) * 80

    return int(dx), int(dy)


def action_to_params(action):
    """Converts a given action index into corresponding shot angle and tap time.
    """
    action = np.unravel_index(action, (ANGLE_RESOLUTION, TAP_TIME_RESOLUTION))

    # 선형 등간격 각도: PHI부터 (180 - PSI)까지 ANGLE_RESOLUTION 단계
    # 간격 = (180 - PHI - PSI) / (ANGLE_RESOLUTION - 1) ≈ 6.8° per step
    alpha = PHI + int(action[0] * (180 - PHI - PSI) / (ANGLE_RESOLUTION - 1))

    tap_time = int(action[1] / TAP_TIME_RESOLUTION * MAXIMUM_TAP_TIME)

    return alpha, tap_time


for i in range(ANGLE_RESOLUTION * TAP_TIME_RESOLUTION):
    alpha, tap_time = action_to_params(i)
    ACTIONS += ["alpha = %.1f °, tap_time = %d ms" % (alpha, tap_time)]


def _find_java_executable():
    """Java 실행 파일 경로를 찾습니다. 시스템 PATH → JAVA_HOME → 일반 설치 경로 순으로 탐색."""
    # 1) PATH에 java가 있으면 바로 사용
    java = shutil.which("java")
    if java:
        return java

    # 2) JAVA_HOME 환경 변수
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home:
        candidate = os.path.join(java_home, "bin", "java.exe")
        if os.path.isfile(candidate):
            return candidate

    # 3) Windows 일반 설치 경로 탐색
    if sys.platform == "win32":
        common_roots = [
            r"C:\Program Files\Java",
            r"C:\Program Files\Eclipse Adoptium",
            r"C:\Program Files\Microsoft",
            r"C:\Program Files\OpenJDK",
        ]
        for root in common_roots:
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root), reverse=True):  # 최신 버전 우선
                candidate = os.path.join(root, entry, "bin", "java.exe")
                if os.path.isfile(candidate):
                    return candidate

    return None


def _find_running_science_birds():
    """실행 중인 Science Birds 프로세스를 반환합니다. 없으면 None."""
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            name = proc.info['name'] or ''
            exe  = proc.info['exe']  or ''
            if 'Science Birds' in name or 'Science Birds' in exe:
                return proc
            # Linux 빌드 실행파일명: 9001.x86_64
            if '9001' in name or '9001' in exe:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def _start_xvfb(display: str = ":99") -> subprocess.Popen:
    """Linux 헤드리스 모드용 가상 디스플레이(Xvfb)를 시작합니다."""
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x720x24"],
        stdout=open(os.devnull, "w"),
        stderr=open(os.devnull, "w"),
    )
    os.environ["DISPLAY"] = display
    time.sleep(0.5)   # Xvfb 초기화 대기
    return proc


def run_science_birds():
    """Starts the Angry Birds simulation software (if it isn't running already).
    """
    print("Starting Science Birds...")

    # ── 이미 실행 중이면 재사용 ──────────────────────────────────
    existing = _find_running_science_birds()
    if existing is not None:
        print(f"  Science Birds 이미 실행 중 (PID: {existing.pid}) — 재사용합니다.")
        return None  # 소유권 없음 → _cleanup에서 종료하지 않음

    # ── 플랫폼별 실행 파일 선택 ─────────────────────────────────
    if sys.platform == "win32":
        game_dir = os.path.abspath("src/envs/ab/Science Birds 0.3.8/")
        exe_path = os.path.join(game_dir, "Science Birds.exe")
        env_vars = None
    elif sys.platform.startswith("linux"):
        game_dir = os.path.abspath(_linux_ab_dir())
        exe_path = os.path.join(game_dir, "9001.x86_64")
        env_vars = dict(os.environ)
        if "DISPLAY" not in env_vars:
            # Xvfb가 없으면 자동 시작
            _start_xvfb(":99")
            env_vars["DISPLAY"] = ":99"
    else:
        print("  [경고] macOS는 미지원입니다. Science Birds가 이미 실행 중이어야 합니다.")
        return None

    if not os.path.isfile(exe_path):
        raise FileNotFoundError(
            f"\n\n[오류] Science Birds 실행 파일을 찾을 수 없습니다.\n"
            f"  예상 경로: {exe_path}\n"
            f"  저장소가 올바르게 클론되었는지 확인하세요.\n"
        )

    proc = subprocess.Popen([exe_path], cwd=game_dir, env=env_vars)
    print(f"  Science Birds 실행됨: {exe_path}")
    return proc   # 소유권 있음 → _cleanup에서 종료함


class AngryBirds(ParallelEnvironment):
    """A wrapper class for the Science Birds environment."""
    NAME = "angry_birds"
    LEVELS = True
    TIME_RELEVANT = False
    WINS_RELEVANT = True

    def __init__(self, num_par_inst):
        if num_par_inst > 1:
            raise ValueError("ERROR: Yet, only one Angry Birds environment is allowed at the same time. "
                             "You tried to initialize %d parallel environments." % num_par_inst)

        super().__init__(num_par_inst, ACTIONS)

        self.id = None
        self.comm_interface = None
        self.observer = None
        self.framework_process = None
        self.science_birds_process = None   # Science Birds 프로세스 핸들
        self._owns_science_birds = False    # 직접 실행했으면 True, 재사용이면 False
        self._cleaned_up = False
        atexit.register(self._cleanup)

        self.validation_levels = []
        self.demo_levels = []
        self.train_levels = list(FILTERED_TRAIN_LEVELS)  # setup_connections()에서 재설정될 수 있음
        self.mode = "train"  # level selection mode: training, testing, validation, demo

        # 새 종류 추적
        self.current_level_birds = []   # 현재 레벨의 새 발사 순서
        self.current_shot_idx    = 0    # 다음에 발사할 새의 인덱스
        self.current_level       = None
        self.last_step_info      = {}

        self.run_framework()
        # Linux 빌드: game_playing_interface.jar가 9001.x86_64를 직접 실행한다.
        # Python이 따로 실행하면 두 번 켜지므로, Linux에서는 건너뛴다.
        if not sys.platform.startswith("linux"):
            proc = run_science_birds()
            if proc is not None:
                self.science_birds_process = proc
                self._owns_science_birds = True
        self.setup_connections()

        self.set_sim_speed(100)

        print("Initialized Angry Birds successfully!")

    def run_framework(self):
        """Starts the server which communicates between Science Birds and the agent."""
        print("Starting the framework...")

        java_exe = _find_java_executable()
        if java_exe is None:
            raise EnvironmentError(
                "\n\n[오류] Java를 찾을 수 없습니다.\n"
                "game_playing_interface.jar 실행에 Java 8 이상이 필요합니다.\n\n"
                "해결 방법:\n"
                "  1) https://adoptium.net/ 에서 Java 설치\n"
                "  2) 설치 후 시스템 환경 변수 PATH에 Java bin 폴더 추가\n"
                "  3) 새 터미널에서 다시 실행\n\n"
                "  확인 명령어: java -version\n"
            )

        if sys.platform.startswith("linux"):
            framework_dir = os.path.abspath(_linux_ab_dir())
            _generate_linux_config(framework_dir)
        else:
            framework_dir = os.path.abspath("src/envs/ab/AB Framework 0.3.8/")
        jar_path = os.path.join(framework_dir, "game_playing_interface.jar")

        if not os.path.isfile(jar_path):
            raise FileNotFoundError(
                f"\n\n[오류] game_playing_interface.jar 파일을 찾을 수 없습니다.\n"
                f"예상 경로: {jar_path}\n"
                f"저장소가 올바르게 클론되었는지 확인하세요.\n"
            )

        java_args = [java_exe, '-jar', 'game_playing_interface.jar']

        log_path = os.path.join(framework_dir, "java_server.log")
        log_file = open(log_path, 'w')
        self.framework_process = subprocess.Popen(
            java_args,
            stdout=log_file,
            stderr=log_file,
            cwd=framework_dir,
        )
        print(f"  Java 서버 시작됨 (PID: {self.framework_process.pid})")
        print(f"  Java 로그: {log_path}")

    def _cleanup(self):
        """Java 서버와 Science Birds 프로세스를 종료합니다. atexit 및 __del__ 중복 호출 방지."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self.framework_process is not None:
            self.framework_process.terminate()
        if self._owns_science_birds and self.science_birds_process is not None:
            try:
                self.science_birds_process.terminate()
            except (psutil.NoSuchProcess, AttributeError, OSError):
                pass
        print("Deleted Angry Birds environment.")

    def _restart_game(self):
        """Unity/Java가 응답 없을 때 게임 프로세스 전체를 재시작하고 소켓을 재연결한다."""
        print("\n[재시작] 게임 응답 없음 — Java+Unity 프로세스를 재시작합니다...", flush=True)

        # 기존 Java(+Unity) 프로세스 강제 종료
        if self.framework_process is not None:
            try:
                self.framework_process.kill()
                self.framework_process.wait(timeout=15)
            except Exception:
                pass
            self.framework_process = None

        # 소켓 닫기
        for client in [self.comm_interface, self.observer]:
            if client is not None:
                try:
                    client.server_socket.close()
                except Exception:
                    pass

        # 포트가 완전히 해제될 때까지 대기
        time.sleep(5)

        # 재시작
        self.run_framework()
        self.setup_connections()
        self.set_sim_speed(100)
        self.current_shot_idx = 0
        print("[재시작] 완료.\n", flush=True)

    def __del__(self):
        self._cleanup()

    def setup_connections(self):
        self.id = 2888

        host = "127.0.0.1"
        self.comm_interface = AgentClient(host, "2004", **SERVER_CLIENT_CONFIG)
        self.observer = AgentClient(host, "2006", **SERVER_CLIENT_CONFIG)

        _CONNECT_TIMEOUT = 60   # 최대 대기 시간 (초)
        _RETRY_INTERVAL  = 2    # 재시도 간격 (초)

        def _connect_with_retry(label, client):
            deadline = time.time() + _CONNECT_TIMEOUT
            while True:
                try:
                    print(f"Connecting {label} to server...")
                    client.connect_to_server()
                    return
                except socket.error:
                    if time.time() >= deadline:
                        raise EnvironmentError(
                            f"\n\n[오류] {_CONNECT_TIMEOUT}초 내에 서버에 연결하지 못했습니다 ({label}).\n"
                            f"Java 서버 또는 Science Birds가 정상 실행 중인지 확인하세요.\n"
                        )
                    print(f"  서버 대기 중... {_RETRY_INTERVAL}초 후 재시도")
                    time.sleep(_RETRY_INTERVAL)
                    try:
                        client.server_socket.close()
                    except OSError:
                        pass
                    client.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client._buffer = bytearray()

        # port 2004 agent 연결 → configure() 호출
        _connect_with_retry("agent (port 2004)", self.comm_interface)
        _round, _limit, level_count = self.comm_interface.configure(self.id)
        print(f"  라운드={_round}, 레벨 수={level_count}")

        # Linux jar: configure()는 TRAINING 모드에서 항상 level_count=0을 반환.
        # readyForNewSet(code 68)을 호출해야 게임이 NEWTRAININGSET 상태를 벗어난다.
        if sys.platform.startswith("linux"):
            print("  readyForNewSet 전송 중...")
            trial_info = self.comm_interface.ready_for_new_set()
            print(f"  readyForNewSet 응답: {trial_info}")
            # 레벨 수는 configure()가 아니라 ready_for_new_set 이후 get_number_of_levels로 확인
            time.sleep(1)
            actual_levels = self.comm_interface.get_number_of_levels()
            print(f"  실제 레벨 수: {actual_levels}")
            if actual_levels > 0:
                # Use the filtered 200-level pool, not all levels the game reports
                self.train_levels = list(FILTERED_TRAIN_LEVELS)
                print(f"  Linux 훈련 레벨: {len(self.train_levels)}개 필터링 레벨 "
                      f"(게임 전체={actual_levels})")
        elif level_count > 0:
            self.train_levels = list(range(1, level_count + 1))

        # port 2006 observer: Linux 버전 jar는 observer 서버를 열지 않음.
        if not sys.platform.startswith("linux"):
            _connect_with_retry("observer (port 2006)", self.observer)
            self.observer.configure(self.id)

    def reset(self, ids=None, **kwargs):
        super(AngryBirds, self).reset(ids)
        if ids is None:
            self._load_next_level_with_recovery()
            self.times[:] = 0
            self.game_overs[:] = False
        else:
            self._load_next_level_with_recovery()  # TODO: multiple envs
            self.times[ids] = 0
            self.game_overs[ids] = False

    def set_mode(self, mode):
        """Sets the environment's level selection mode. There are four options:
         - 'train': selects non-validation levels randomly
         - 'test': selects any level randomly
         - 'validate': selects only validation levels
         - 'demo': selects only demo levels"""
        if mode in ["train", "test", "validate", "demo"]:
            self.mode = mode
        else:
            raise ValueError("ERROR: Invalid mode option given. You provided %s but only "
                             "'train', 'test', 'validate', and 'demo' are allowed." % str(mode))

    def _load_next_level_with_recovery(self, max_attempts=2):
        for attempt in range(max_attempts):
            try:
                self.load_next_level()
                return
            except COMMUNICATION_ERRORS as exc:
                print(
                    f"\n[오류] 레벨 로드/리셋 통신 실패 "
                    f"({exc.__class__.__name__}): {exc}",
                    flush=True,
                )
                if attempt >= max_attempts - 1:
                    raise
                self._restart_game()

    def load_next_level(self):
        """Loads a level, depending on the level selection mode."""

        if self.mode in ("train", "test"):
            next_level = np.random.choice(self.train_levels)

        elif self.mode == "validate":
            next_level = np.random.choice(self.validation_levels)

        else:
            next_level = np.random.choice(self.demo_levels)

        next_level = int(next_level)
        self.current_level = next_level
        self.comm_interface.load_level(next_level)

        if sys.platform.startswith("linux"):
            # load_level() 수락 후 실제 PLAYING 전환까지 대기.
            # REQUESTNOVELTYLIKELIHOOD 상태에서는 반드시 report_novelty_likelihood()를
            # 보내야 PLAYING으로 전환된다 (경쟁 모드 프로토콜).
            _PLAYABLE = {GameState.PLAYING, GameState.WON, GameState.LOST}
            deadline = time.time() + 30
            while time.time() < deadline:
                state = self.comm_interface.get_game_state()
                if state in _PLAYABLE:
                    break
                if state == GameState.REQUESTNOVELTYLIKELIHOOD:
                    self.comm_interface.report_novelty_likelihood(0.0)
                    time.sleep(0.3)
                    continue
                time.sleep(0.5)

        self.current_level_birds = LEVEL_BIRD_MAP.get(next_level, [])
        self.current_shot_idx    = 0

    def load_specified_level(self, level_number=None):
        self.current_level = int(level_number) if level_number is not None else None
        self.comm_interface.load_level(level_number)
        self.current_level_birds = LEVEL_BIRD_MAP.get(level_number, [])
        self.current_shot_idx    = 0

    def step(self, actions):
        action_idx = int(np.asarray(actions).reshape(-1)[0])
        alpha, tap_time = action_to_params(action_idx)
        score_before = int(self.scores[0])
        _, score, appl_state = self.perform_actions(action_idx)
        score = np.array([score], dtype='uint')
        won  = (appl_state == GameState.WON)
        lost = (appl_state == GameState.LOST)

        # On Linux, competition-mode protocol: after WON/LOST Unity immediately
        # enters REQUESTNOVELTYLIKELIHOOD before the agent can see WON/LOST.
        # Any non-playable state that isn't explicitly WON/LOST means level over.
        if sys.platform.startswith("linux"):
            _PLAYABLE = {GameState.PLAYING, GameState.UNSTABLE}
            if appl_state not in _PLAYABLE and not won and not lost:
                lost = True

        game_over = won or lost

        # ── Reward ────────────────────────────────────────────
        # ORIGINAL:
        # reward = score2reward(score)   # 점수 기반 기본 보상
        # if won:
        #     reward += 5.0             # 레벨 클리어 보너스
        # elif lost:
        #     reward -= 0.5             # 실패 페널티 (탐험 유도)

        # UPDATED: score-delta reward.  This is the only changed behavior in
        # the environment wrapper; observation/action/server interaction stays
        # untouched for final-evaluation compatibility.
        score_after = int(score[0])
        score_delta = score_after - score_before
        score_reward = score_delta / float(SCORE_NORMALIZATION)
        win_bonus = WIN_BONUS if won else 0.0
        loss_penalty = LOSS_PENALTY if lost else 0.0
        shot_penalty = SHOT_PENALTY
        reward = np.array([score_reward], dtype="float32")
        if won:
            reward += WIN_BONUS
        if lost:
            reward -= LOSS_PENALTY
        reward -= SHOT_PENALTY
        if USE_REWARD_CLIP:
            reward = np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)

        self.last_step_info = {
            "level": self.current_level,
            "shot_idx": int(self.current_shot_idx),
            "action": action_idx,
            "angle": int(alpha),
            "tap_ms": int(tap_time),
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_delta,
            "score_reward": float(score_reward),
            "win_bonus": float(win_bonus),
            "loss_penalty": float(loss_penalty),
            "shot_penalty": float(shot_penalty),
            "final_reward": float(reward[0]),
            "won": bool(won),
            "lost": bool(lost),
            "game_over": bool(game_over),
            "app_state": str(appl_state),
        }

        self.scores[:] = score
        self.times += 1
        self.game_overs[:] = game_over
        self.wins[:] = won            # 클리어 여부 기록
        return reward, score, self.game_overs, self.times, self.wins, self.game_overs

    def _make_empty_state(self):
        image_state = np.zeros((1, STATE_PIXEL_RES, STATE_PIXEL_RES, 3), dtype=np.uint8)
        bird_vec = np.zeros(BIRD_DIM, dtype=np.float32)
        return [image_state, np.tile(bird_vec, (self.num_par_inst, 1))]

    def perform_actions(self, action):
        """Performs a shot and observes and returns the consequences."""
        try:
            return self._perform_actions_inner(action)
        except COMMUNICATION_ERRORS as exc:
            print(f"\n[오류] 게임 통신 실패 ({exc.__class__.__name__}): {exc}", flush=True)
            self._restart_game()
            return self._make_empty_state(), 0, GameState.LOST

    def _perform_actions_inner(self, action):
        # Convert action index into aim vector and tap time
        alpha, tap_time = action_to_params(action)

        # Linux: check game state BEFORE shooting.
        # After a winning shot, Unity can jump straight from WON to
        # REQUESTNOVELTYLIKELIHOOD before Python gets to read the state.
        # REQUESTNOVELTYLIKELIHOOD also appears transiently between birds within
        # a multi-bird level.  In both cases we must acknowledge it first
        # (report_novelty_likelihood), then wait for PLAYING.
        # Only truly terminal states (WON/LOST) cause an early return.
        if sys.platform.startswith("linux"):
            _PLAYABLE = {GameState.PLAYING, GameState.UNSTABLE}
            _TERMINAL = {GameState.WON, GameState.LOST}
            deadline = time.time() + 10
            while time.time() < deadline:
                pre_state = self.comm_interface.get_game_state()
                if pre_state in _PLAYABLE:
                    break
                if pre_state in _TERMINAL:
                    return self._make_empty_state(), 0, pre_state
                if pre_state == GameState.REQUESTNOVELTYLIKELIHOOD:
                    self.comm_interface.report_novelty_likelihood(0.0)
                    time.sleep(0.2)
                    continue
                time.sleep(0.2)
            else:
                # Timed out — treat as terminal so the training loop can reset.
                return self._make_empty_state(), 0, GameState.LOST

        # Perform the shot
        sling_x = 214
        sling_y = 356
        rad = np.deg2rad(alpha)
        pull_x = sling_x - int(np.sin(rad) * 80)
        pull_y = sling_y + int(np.cos(rad) * 80)
        self.comm_interface.shoot(pull_x, pull_y, 0, tap_time, 0, 0, isPolar=False)

        # 발사 완료 → 다음 새로 인덱스 이동
        self.current_shot_idx += 1

        # Get the environment state (cropped screenshot)
        env_state = self.get_states()

        # Obtain game score
        score = self.comm_interface.get_current_score()

        # Get the application state.
        # On Linux, REQUESTNOVELTYLIKELIHOOD can appear transiently even in the
        # middle of a level (competition-mode protocol).  If we see it here,
        # acknowledge it and wait for a stable state before returning so that
        # step() can accurately decide whether the level is over.
        appl_state = self.comm_interface.get_game_state()
        if sys.platform.startswith("linux") and appl_state == GameState.REQUESTNOVELTYLIKELIHOOD:
            _TERMINAL = {GameState.WON, GameState.LOST, GameState.PLAYING, GameState.UNSTABLE}
            deadline = time.time() + 10
            while time.time() < deadline:
                self.comm_interface.report_novelty_likelihood(0.0)
                time.sleep(0.2)
                next_state = self.comm_interface.get_game_state()
                if next_state in _TERMINAL or next_state != GameState.REQUESTNOVELTYLIKELIHOOD:
                    appl_state = next_state
                    break

        return env_state, score, appl_state

    def get_states(self):
        # 레벨 전환 직후 Unity 오브젝트가 초기화되지 않아
        # NullReferenceException이 발생할 수 있으므로 재시도 로직을 사용한다.
        max_retries = 5
        retry_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                # Linux jar는 GetGroundTruthWithScreenshot(61)를 지원하지 않아 블록됨.
                # do_screenshot()(code 11)은 이미지만 반환하며 ground_truth는
                # LEVEL_BIRD_MAP에서 읽으므로 실시간 ground truth가 불필요하다.
                if sys.platform.startswith("linux"):
                    screenshot = self.comm_interface.do_screenshot()
                else:
                    screenshot, _ = self.comm_interface.get_ground_truth_with_screenshot()

                # Crop the image to reduce information overload.
                # The cropped image has then dimension (325, 800, 3).
                crop = screenshot[75:400, 40:]

                # Rescale the image into a (smaller) square
                scaled = cv2.resize(crop, (STATE_PIXEL_RES, STATE_PIXEL_RES))

                # Image state: shape (num_par_inst, H, W, C)
                image_state = np.expand_dims(scaled.astype(np.uint8), axis=0)
                break  # 성공

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  [경고] 스크린샷 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    print(f"  레벨 로딩 중일 수 있습니다. {retry_delay:.0f}초 후 재시도...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 8.0)  # 최대 8초
                else:
                    print(f"  [경고] 스크린샷 {max_retries}회 실패, 빈 화면으로 대체합니다.")
                    image_state = np.zeros(
                        (1, STATE_PIXEL_RES, STATE_PIXEL_RES, 3), dtype=np.uint8
                    )
        bird_vec = np.zeros(BIRD_DIM, dtype=np.float32)
        if self.current_level_birds and self.current_shot_idx < len(self.current_level_birds):
            bird_type = self.current_level_birds[self.current_shot_idx]
            idx = BIRD_TYPE_IDX.get(bird_type)
            if idx is not None:
                bird_vec[idx] = 1.0
        # shape: (num_par_inst, BIRD_DIM)
        numerical_state = np.tile(bird_vec, (self.num_par_inst, 1))

        return [image_state, numerical_state]

    def get_state_shapes(self):
        image_state_shape   = (STATE_PIXEL_RES, STATE_PIXEL_RES, 3)
        numerical_state_shape = (BIRD_DIM,)   # (5,)
        return [image_state_shape, numerical_state_shape]

    def get_state_dtypes(self):
        return [np.dtype(np.uint8), np.dtype(np.float32)]

    def preprocess(self, states):
        """
        AngryBirds 상태 전처리:

        """
        image_state = states[0].astype("float32") / 255.0
        bird_state  = states[1].astype("float32")
        return [image_state, bird_state]

    def get_number_of_actions(self):
        return len(self.actions)

    def set_sim_speed(self, speed):
        self.comm_interface.set_game_simulation_speed(speed)

    def get_last_step_diagnostics(self):
        return dict(self.last_step_info)

    def get_diagnostics_config(self):
        return {
            "state_pixel_res": STATE_PIXEL_RES,
            "bird_types": list(BIRD_TYPES),
            "bird_dim": BIRD_DIM,
            "angle_resolution": ANGLE_RESOLUTION,
            "tap_time_resolution": TAP_TIME_RESOLUTION,
            "maximum_tap_time": MAXIMUM_TAP_TIME,
            "phi": PHI,
            "psi": PSI,
            "num_train_levels": len(self.train_levels),
            "train_levels": list(map(int, self.train_levels)),
            "reward": {
                "score_normalization": SCORE_NORMALIZATION,
                "win_bonus": WIN_BONUS,
                "loss_penalty": LOSS_PENALTY,
                "shot_penalty": SHOT_PENALTY,
                "use_reward_clip": USE_REWARD_CLIP,
                "reward_clip_min": REWARD_CLIP_MIN,
                "reward_clip_max": REWARD_CLIP_MAX,
            },
        }


def score2reward(score):
    """Turns scores into rewards."""
    reward = score / SCORE_NORMALIZATION
    return reward
