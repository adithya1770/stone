monitor = TelemetryMonitor()

for _ in range(10):
    print(monitor.get_telemetry())