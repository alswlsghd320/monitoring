#!/bin/bash

# 서비스 설정
SERVICE_NAME="system_monitoring"
PID_FILE="/var/run/$SERVICE_NAME.pid"
BASE_DIR="/var/log/system_monitoring"
RETENTION_DAYS=60

# 필수 디렉토리 생성
mkdir -p $BASE_DIR

# 네트워크 인터페이스 감지 (lo 제외)
NETWORK_INTERFACES=$(ip -o link show | awk -F': ' '{print $2}' | grep -v 'lo')

# 로그 파일 초기화
initialize_logs() {
    local current_date=$1
    MONITOR_DIR="$BASE_DIR/$current_date"
    mkdir -p $MONITOR_DIR
    
    # 로그 파일 경로 설정
    USER_LOG="$MONITOR_DIR/user_monitoring_${current_date}.log"
    RESOURCE_LOG="$MONITOR_DIR/resource_monitoring_${current_date}.log"
    PROCESS_LOG="$MONITOR_DIR/process_monitoring_${current_date}.log"
    NETWORK_LOG="$MONITOR_DIR/network_monitoring_${current_date}.log"
    REPORT_FILE="$MONITOR_DIR/system_report_${current_date}.md"
    DAILY_SUMMARY="$MONITOR_DIR/daily_summary_${current_date}.md"
}

# 오래된 로그 정리
cleanup_old_logs() {
    find $BASE_DIR -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;
}

# 사용자 모니터링
monitor_users() {
    local current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "=== $current_time ===" >> $USER_LOG
    echo "현재 접속자 상태:" >> $USER_LOG
    w >> $USER_LOG
    echo "접속 사용자 수: $(who | wc -l)" >> $USER_LOG
    echo "사용자별 프로세스 수:" >> $USER_LOG
    ps -eo user | sort | uniq -c >> $USER_LOG
    echo "상세 로그인 정보:" >> $USER_LOG
    last | head -n 10 >> $USER_LOG
    echo "-----------------" >> $USER_LOG
}

# 시스템 리소스 모니터링
monitor_resources() {
    local current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "=== $current_time ===" >> $RESOURCE_LOG
    
    echo "CPU 사용률:" >> $RESOURCE_LOG
    mpstat 1 1 >> $RESOURCE_LOG
    
    echo "메모리 상태:" >> $RESOURCE_LOG
    free -m >> $RESOURCE_LOG
    vmstat -s >> $RESOURCE_LOG
    
    echo "로드 애버리지:" >> $RESOURCE_LOG
    uptime >> $RESOURCE_LOG
    
    echo "디스크 사용량:" >> $RESOURCE_LOG
    df -h >> $RESOURCE_LOG
    
    echo "디스크 I/O 상태:" >> $RESOURCE_LOG
    iostat -x 1 1 >> $RESOURCE_LOG
    
    echo "스왑 사용량:" >> $RESOURCE_LOG
    swapon -s >> $RESOURCE_LOG
    
    echo "-----------------" >> $RESOURCE_LOG
}

# 프로세스 모니터링
monitor_processes() {
    local current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "=== $current_time ===" >> $PROCESS_LOG
    
    echo "CPU 사용률 Top 10 프로세스:" >> $PROCESS_LOG
    ps aux --sort=-%cpu | head -11 >> $PROCESS_LOG
    
    echo "메모리 사용률 Top 10 프로세스:" >> $PROCESS_LOG
    ps aux --sort=-%mem | head -11 >> $PROCESS_LOG
    
    echo "좀비 프로세스:" >> $PROCESS_LOG
    ps aux | grep Z >> $PROCESS_LOG
    
    echo "실행 시간이 긴 프로세스:" >> $PROCESS_LOG
    ps -eo pid,user,pcpu,pmem,time,comm --sort=-time | head -6 >> $PROCESS_LOG
    
    echo "-----------------" >> $PROCESS_LOG
}

# 네트워크 모니터링
monitor_network() {
    local current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "=== $current_time ===" >> $NETWORK_LOG
    
    echo "네트워크 인터페이스 상태:" >> $NETWORK_LOG
    for interface in $NETWORK_INTERFACES; do
        echo "[ $interface ]" >> $NETWORK_LOG
        ip -s link show $interface >> $NETWORK_LOG
        
        echo "대역폭 사용량:" >> $NETWORK_LOG
        sar -n DEV 1 5 | grep $interface >> $NETWORK_LOG
    done
    
    echo "TCP 연결 상태:" >> $NETWORK_LOG
    ss -s >> $NETWORK_LOG
    
    echo "현재 네트워크 연결:" >> $NETWORK_LOG
    netstat -ant | awk '{print $6}' | sort | uniq -c >> $NETWORK_LOG
    
    echo "네트워크 에러 통계:" >> $NETWORK_LOG
    netstat -s | grep -i error >> $NETWORK_LOG
    
    echo "-----------------" >> $NETWORK_LOG
}

# 일일 보고서 생성
generate_report() {
    local current_date=$1
    local monitor_dir="$BASE_DIR/$current_date"
    local report_file="$monitor_dir/system_report_${current_date}.md"
    
    cat << EOF > $report_file
# 시스템 상세 분석 보고서

## 1. 모니터링 정보
- 분석 일자: ${current_date:0:4}년 ${current_date:4:2}월 ${current_date:6:2}일
- 보고서 생성 시간: $(date +"%Y-%m-%d %H:%M:%S")

## 2. 시스템 사양
\`\`\`
$(lscpu | grep "Model name\|CPU(s):")
$(free -h | head -2)
$(df -h /)
\`\`\`

## 3. 시스템 부하 분석
### CPU 사용률
- 최대: $(grep "all" $RESOURCE_LOG | awk '{print $3}' | sort -nr | head -1)%
- 평균: $(grep "all" $RESOURCE_LOG | awk '{sum+=$3} END {print sum/NR}')%

### 메모리 사용률
- 사용 중: $(free -m | awk 'NR==2 {printf "%.2f%%", $3*100/$2}')
- 가용: $(free -m | awk 'NR==2 {printf "%.2f%%", $4*100/$2}')

### 디스크 I/O
$(iostat -x | grep "^[s|v]d[a-z]" | head -3)

## 4. 사용자 활동
- 최대 동시접속: $(grep "접속 사용자 수:" $USER_LOG | awk '{print $4}' | sort -nr | head -1)명
- 현재 접속자: $(who | wc -l)명

## 5. 네트워크 상태
$(for interface in $NETWORK_INTERFACES; do
    echo "### $interface"
    echo "- 수신: $(grep "$interface" $NETWORK_LOG | awk '/RX:/ {sum+=$3} END {printf "%.2f GB", sum/1024/1024/1024}')"
    echo "- 송신: $(grep "$interface" $NETWORK_LOG | awk '/TX:/ {sum+=$3} END {printf "%.2f GB", sum/1024/1024/1024}')"
done)

## 6. 주요 이벤트
- CPU 과부하(>80%): $(grep "all" $RESOURCE_LOG | awk '$3>80 {count++} END {print count}')회
- 메모리 부족(<10%): $(grep "Mem:" $RESOURCE_LOG | awk '{if($4/$2<0.1) count++} END {print count}')회

## 7. 조치 필요 사항
EOF

    # 조치사항 추가
    local cpu_high=$(grep "all" $RESOURCE_LOG | awk '$3>80 {count++} END {print count}')
    local mem_low=$(grep "Mem:" $RESOURCE_LOG | awk '{if($4/$2<0.1) count++} END {print count}')
    
    [ $cpu_high -gt 0 ] && echo "- CPU 사용률 개선 필요" >> $report_file
    [ $mem_low -gt 0 ] && echo "- 메모리 증설 검토" >> $report_file
}

# 일간 요약 보고서 생성
generate_daily_summary() {
    local current_date=$1
    local monitor_dir="$BASE_DIR/$current_date"
    local daily_summary="$monitor_dir/daily_summary_${current_date}.md"
    
    cat << EOF > $daily_summary
# 일간 시스템 모니터링 요약
날짜: ${current_date:0:4}년 ${current_date:4:2}월 ${current_date:6:2}일

## 1. 핵심 지표 요약
### 시스템 부하
- CPU 평균: $(grep "all" $RESOURCE_LOG | awk '{sum+=$3} END {printf "%.1f%%", sum/NR}')
- 메모리 사용률: $(free -m | awk 'NR==2 {printf "%.1f%%", $3*100/$2}')
- 디스크 사용률: $(df -h / | awk 'NR==2 {print $5}')

### 접속자 통계
- 최대: $(grep "접속 사용자 수:" $USER_LOG | awk '{print $4}' | sort -nr | head -1)명
- 평균: $(grep "접속 사용자 수:" $USER_LOG | awk '{sum+=$4} END {printf "%.1f", sum/NR}')명

### 네트워크 트래픽
$(for interface in $NETWORK_INTERFACES; do
    echo "- $interface:"
    echo "  - 수신: $(grep "$interface" $NETWORK_LOG | awk '/RX:/ {sum+=$3} END {printf "%.2f GB", sum/1024/1024/1024}')"
    echo "  - 송신: $(grep "$interface" $NETWORK_LOG | awk '/TX:/ {sum+=$3} END {printf "%.2f GB", sum/1024/1024/1024}')"
done)

## 2. 주요 이벤트
- 과부하 발생: $(grep "all" $RESOURCE_LOG | awk '$3>80 {count++} END {print count}')회
- 메모리 부족: $(grep "Mem:" $RESOURCE_LOG | awk '{if($4/$2<0.1) count++} END {print count}')회
- 디스크 I/O 대기: $(iostat -x | awk '$10>10 {count++} END {print count}')회

## 3. 상위 프로세스
### CPU 사용률 Top 3
\`\`\`
$(ps aux --sort=-%cpu | head -4)
\`\`\`

### 메모리 사용률 Top 3
\`\`\`
$(ps aux --sort=-%mem | head -4)
\`\`\`

## 4. 권장 조치사항
EOF

    # 권장 조치사항 추가
    local cpu_high=$(grep "all" $RESOURCE_LOG | awk '$3>80 {count++} END {print count}')
    local mem_low=$(grep "Mem:" $RESOURCE_LOG | awk '{if($4/$2<0.1) count++} END {print count}')
    local disk_io_high=$(iostat -x | awk '$10>10 {count++} END {print count}')
    
    [ $cpu_high -gt 0 ] && echo "- CPU 사용률 개선 검토" >> $daily_summary
    [ $mem_low -gt 0 ] && echo "- 메모리 증설 검토" >> $daily_summary
    [ $disk_io_high -gt 0 ] && echo "- 디스크 I/O 최적화 필요" >> $daily_summary
}

# 서비스 시작
start() {
    if [ -f "$PID_FILE" ]; then
        echo "서비스가 이미 실행 중입니다."
        exit 1
    fi
    
    {
        echo $$ > $PID_FILE
        cleanup_old_logs
        
        while true; do
            CURRENT_DATE=$(date +"%Y%m%d")
            
            if [ ! -d "$BASE_DIR/$CURRENT_DATE" ]; then
                if [ -n "$TODAY" ]; then
                    generate_report $TODAY
                    generate_daily_summary $TODAY
                fi
                
                initialize_logs $CURRENT_DATE
                TODAY=$CURRENT_DATE
            fi
            
            monitor_users
            monitor_resources
            monitor_processes
            monitor_network
            
            if [ "$(date +%M)" == "00" ]; then
                generate_report $CURRENT_DATE
                generate_daily_summary $CURRENT_DATE
            fi
            
            sleep 300
        done
    } &
    
    echo "시스템 모니터링이 시작되었습니다. (PID: $!)"
}

# 서비스 중지
stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "서비스가 실행 중이지 않습니다."
        exit 1
    fi
    
    PID=$(cat $PID_FILE)
    kill -15 $PID
    rm -f $PID_FILE
    echo "시스템 모니터링이 중지되었습니다."
}

# 서비스 상태 확인
status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ps -p $PID > /dev/null; then
            echo "서비스가 실행 중입니다. (PID: $PID)"
        else
            echo "서비스가 비정상 종료되었습니다."
            rm -f $PID_FILE
        fi
    else
        echo "서비스가 실행 중이지 않습니다."
    fi
}

# 명령어 처리
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    *)
        echo "사용법: $0
