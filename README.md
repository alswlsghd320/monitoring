# 우분투 시스템 모니터링 가이드

## 1. 개요
이 문서는 Ubuntu Linux 환경에서 시스템 리소스, 사용자 활동, 네트워크 상태를 모니터링하고 일간 보고서를 생성하는 방법을 설명합니다.

## 2. 기능
- CPU, 메모리, 디스크 사용량 모니터링
- 사용자 접속 및 활동 모니터링
- 네트워크 트래픽 및 연결 상태 모니터링
- 프로세스 모니터링
- 일간 요약 보고서 자동 생성
- 60일간 로그 보관
- systemd 서비스로 실행

## 3. 설치 방법

### 3.1 필요한 패키지 설치
```bash
# 필요한 시스템 도구 설치
sudo apt update
sudo apt install -y sysstat net-tools bc htop
```

### 3.2 sysstat 활성화
```bash
# sysstat 서비스 활성화
sudo systemctl enable sysstat
sudo systemctl start sysstat

# sysstat 설정 수정 (필요한 경우)
sudo vi /etc/default/sysstat
# ENABLED="true" 확인
```

### 3.3 모니터링 스크립트 설치
```bash
# 스크립트 저장
sudo vi /usr/local/bin/system_monitor

# 실행 권한 부여
sudo chmod +x /usr/local/bin/system_monitor

# 로그 디렉토리 생성
sudo mkdir -p /var/log/system_monitoring
sudo chmod 755 /var/log/system_monitoring
```

### 3.4 systemd 서비스 설정
```bash
# 서비스 파일 생성
sudo vi /etc/systemd/system/system_monitor.service

[Unit]
Description=System Monitoring Service
After=network.target

[Service]
Type=forking
ExecStart=/usr/local/bin/system_monitor start
ExecStop=/usr/local/bin/system_monitor stop
PIDFile=/run/system_monitoring.pid
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3.5 서비스 활성화
```bash
sudo systemctl daemon-reload
sudo systemctl enable system_monitor
sudo systemctl start system_monitor
```

## 4. 디렉토리 구조
```
/var/log/system_monitoring/
├── YYYYMMDD/
│   ├── user_monitoring_YYYYMMDD.log    # 사용자 활동 로그
│   ├── resource_monitoring_YYYYMMDD.log # 시스템 리소스 로그
│   ├── process_monitoring_YYYYMMDD.log  # 프로세스 로그
│   ├── network_monitoring_YYYYMMDD.log  # 네트워크 로그
│   ├── system_report_YYYYMMDD.md       # 상세 보고서
│   └── daily_summary_YYYYMMDD.md       # 일간 요약 보고서
```

## 5. 모니터링 내용

### 5.1 사용자 모니터링
- 현재 접속자 수 및 정보
- 사용자별 프로세스 수
- SSH 접속 기록

### 5.2 시스템 리소스 모니터링
- CPU 사용률 및 로드 애버리지
- 메모리 및 스왑 사용량
- 디스크 사용량 및 I/O 통계
- 시스템 온도 (가능한 경우)

### 5.3 프로세스 모니터링
- CPU/메모리 사용률 상위 프로세스
- 좀비 프로세스 확인
- 실행 시간이 긴 프로세스

### 5.4 네트워크 모니터링
- 인터페이스별 트래픽
- 연결 상태
- 네트워크 에러

## 6. 사용 방법

### 6.1 서비스 제어
```bash
# 서비스 상태 확인
sudo systemctl status system_monitor

# 서비스 시작/중지/재시작
sudo systemctl start system_monitor
sudo systemctl stop system_monitor
sudo systemctl restart system_monitor
```

### 6.2 로그 확인
```bash
# 오늘의 보고서 확인
cat /var/log/system_monitoring/$(date +%Y%m%d)/daily_summary_$(date +%Y%m%d).md

# 실시간 로그 모니터링
tail -f /var/log/system_monitoring/$(date +%Y%m%d)/resource_monitoring_$(date +%Y%m%d).log
```

### 6.3 문제 해결
```bash
# 서비스 로그 확인
journalctl -u system_monitor -n 50

# 권한 확인
ls -l /usr/local/bin/system_monitor
ls -l /var/log/system_monitoring

# 프로세스 확인
ps aux | grep system_monitor
```

## 7. 보안 고려사항
- 로그 파일 권한 관리
  ```bash
  sudo chown -R root:root /var/log/system_monitoring
  sudo chmod -R 755 /var/log/system_monitoring
  ```
- AppArmor 프로필 설정 (선택사항)
- 민감한 프로세스 정보 제외 설정

## 8. 유지보수

### 8.1 로그 로테이션
- 60일 이상된 로그 자동 삭제
- 디스크 공간 모니터링

### 8.2 성능 최적화
- 모니터링 간격 조정 (기본 5분)
- 리소스 사용량이 높은 경우 간격 증가

## 9. 알림 설정 (선택사항)
```bash
# 이메일 알림을 위한 mailutils 설치
sudo apt install -y mailutils

# Discord/Slack 웹훅 설정
# 스크립트 내 WEBHOOK_URL 변수 설정
```

## 10. 추가 도구 연동
- Prometheus 메트릭 출력
- Grafana 대시보드 연동
- 로그 집계 시스템(ELK) 연동

## 11. 트러블슈팅
- 서비스 시작 실패: SELinux/AppArmor 설정 확인
- 로그 생성 실패: 디스크 공간 및 권한 확인
- 높은 CPU 사용률: 모니터링 간격 조정

## 12. 참고사항
- Ubuntu 20.04 LTS 이상 지원
- 최소 시스템 요구사항: 512MB RAM, 1GB 디스크 공간
- python3 기반 확장 가능
