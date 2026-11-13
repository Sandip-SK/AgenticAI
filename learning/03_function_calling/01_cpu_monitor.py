
import psutil
import time

def get_cpu_usage():
    cpu = psutil.cpu_percent(interval=1)

    return {
        "cpu_usage": cpu,
        "status": "High" if cpu > 80 else "Normal"
    }

result = get_cpu_usage()

print(result)
