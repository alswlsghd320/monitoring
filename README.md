# Rocky Linux 시스템 모니터링 가이드

## 1. 개요
이 문서는 Rocky Linux 환경에서 시스템 리소스, 사용자 활동, 네트워크 상태를 모니터링하고 일간 보고서를 생성하는 방법을 설명합니다.

## 2. 기능
- CPU, 메모리, 디스크 사용량 모니터링
- 사용자 접속 및 활동 모니터링
- 네트워크 트래픽 및 연결 상태 모니터링
- 프로세스 모니터링
- 일간 요약 보고서 자동 생성
- 60일간 로그 보관
- 시스템 서비스로 실행 가능

## 3. 설치 방법

### 3.1 필요한 패키지 설치
```bash
# 필요한 시스템 도구 설치
sudo yum install -y sysstat net-tools bc
```

### 3.2 모니터링 스크립트 설치
```bash
# 스크립트 저장
sudo vi /usr/local/bin/system_monitor

# 실행 권한 부여
sudo chmod +x /usr/local/bin/system_monitor

# 로그 디렉토리 생성
sudo mkdir -p /var/log/system_monitoring
sudo chmod 755 /var/log/system_monitoring
```

### 3.3 systemd 서비스 설정
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
PIDFile=/var/run/system_monitoring.pid
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3.4 서비스 활성화
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
- 접속 시간 및 IP 정보

### 5.2 시스템 리소스 모니터링
- CPU 사용률
- 메모리 사용량
- 스왑 사용량
- 디스크 사용량
- 로드 애버리지

### 5.3 프로세스 모니터링
- CPU 사용률 Top 10 프로세스
- 메모리 사용률 Top 10 프로세스
- 사용자별 프로세스 통계

### 5.4 네트워크 모니터링
- 인터페이스별 트래픽 통계
- 대역폭 사용량
- TCP/UDP 연결 상태
- 네트워크 에러 통계

## 6. 보고서 종류

### 6.1 시스템 상세 보고서 (system_report_YYYYMMDD.md)
- 상세한 시스템 상태 정보
- 시간별 리소스 사용량
- 문제점 분석
- 개선 제안사항

### 6.2 일간 요약 보고서 (daily_summary_YYYYMMDD.md)
- 하루 동안의 주요 지표 요약
- 리소스 사용량 최대/평균값
- 주요 이벤트 기록
- 권장 조치사항

## 7. 사용 방법

### 7.1 서비스 제어
```bash
# 서비스 상태 확인
sudo systemctl status system_monitor

# 서비스 시작
sudo systemctl start system_monitor

# 서비스 중지
sudo systemctl stop system_monitor

# 서비스 재시작
sudo systemctl restart system_monitor
```

### 7.2 로그 확인
```bash
# 오늘의 상세 보고서 확인
cat /var/log/system_monitoring/$(date +%Y%m%d)/system_report_$(date +%Y%m%d).md

# 오늘의 요약 보고서 확인
cat /var/log/system_monitoring/$(date +%Y%m%d)/daily_summary_$(date +%Y%m%d).md

# 특정 날짜의 네트워크 로그 확인
cat /var/log/system_monitoring/20240124/network_monitoring_20240124.log

# 실시간 로그 모니터링
tail -f /var/log/system_monitoring/$(date +%Y%m%d)/resource_monitoring_$(date +%Y%m%d).log
```

### 7.3 로그 관리
- 로그는 60일간 보관됨
- 60일 이상된 로그는 자동으로 삭제됨
- 각 날짜별로 별도 디렉토리에 저장

## 8. 문제 해결

### 8.1 서비스가 시작되지 않는 경우
```bash
# 로그 확인
journalctl -u system_monitor -n 50

# 권한 확인
ls -l /usr/local/bin/system_monitor
ls -l /var/log/system_monitoring

# PID 파일 확인
ls -l /var/run/system_monitoring.pid
```

### 8.2 로그가 생성되지 않는 경우
```bash
# 디스크 공간 확인
df -h

# 권한 확인
ls -l /var/log/system_monitoring

# SELinux 상태 확인
getenforce
```

## 9. 보안 고려사항
- 로그 파일은 root 사용자만 접근 가능하도록 설정
- 민감한 정보가 포함된 프로세스는 모니터링에서 제외 가능
- 네트워크 모니터링 시 개인정보 보호 준수

## 10. 권장 사항
- 정기적으로 보고서 확인
- 시스템 리소스 부족 시 즉시 조치
- 필요한 경우 모니터링 간격 조정 (기본 5분)
- 중요 이벤트 발생 시 알림 설정 추가

## 11. 갱신 이력
- 2024-01-24: 최초 작성
- 2024-01-24: 네트워크 모니터링 추가
- 2024-01-24: 일간 요약 보고서 추가
- 2024-01-24: 로그 보관 기간 60일로 연장
