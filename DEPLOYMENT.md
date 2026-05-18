# ORB Live — VPS Deployment Guide

## 1. System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU      | 1 vCPU  | 2 vCPU      |
| RAM      | 1 GB    | 2 GB        |
| Disk     | 10 GB   | 20 GB SSD   |
| OS       | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Python   | 3.11+   | 3.12        |
| Network  | 100 Mbps| 1 Gbps (low-latency) |

Alpaca's WebSocket feed requires a stable outbound connection to
`stream.data.alpaca.markets`. AWS `us-east-1`, Vultr Newark, or Linode
Newark are good choices for US market latency.

---

## 2. First-Time Setup

```bash
# 1. Create deploy user
adduser orb
usermod -aG sudo orb
su - orb

# 2. Install Python
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential

# 3. Clone or copy the project
git clone <your-repo> /opt/orb-live
# or: scp -r ./BacktestingGaps orb@<vps-ip>:/opt/orb-live
cd /opt/orb-live

# 4. Create virtual environment and install dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .   # installs from pyproject.toml

# 5. Create data directories
mkdir -p data/archive data/logs

# 6. Verify installation
python -c "import orb_live; print('OK')"
python -m pytest orb_live/tests/ -q   # should be 150 passed
```

---

## 3. Environment Variables

Create `/opt/orb-live/.env` (never commit this file):

```bash
# Alpaca credentials (live trading)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=false          # set to "true" for paper trading

# Alert webhook (Discord/Telegram/Slack)
ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/...
ALERT_LEVEL=WARN            # INFO | WARN | CRITICAL
ALERT_INFO_OPT_IN=false

# Backup destination (Backblaze B2 / Cloudflare R2 / AWS S3)
BACKUP_BUCKET_URL=b2://my-bucket/orb-live/
BACKUP_ACCESS_KEY=your_b2_key
BACKUP_SECRET_KEY=your_b2_secret

# Health server port (default 8080)
HEALTH_PORT=8080
```

Load in the systemd service with `EnvironmentFile=/opt/orb-live/.env`.

---

## 4. Systemd Service

Create `/etc/systemd/system/orb-live.service`:

```ini
[Unit]
Description=ORB Live Trading Session
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=orb
WorkingDirectory=/opt/orb-live
EnvironmentFile=/opt/orb-live/.env
ExecStart=/opt/orb-live/.venv/bin/python -m orb_live.runner.main
Restart=on-failure
RestartSec=30
StandardOutput=append:/opt/orb-live/data/logs/session.log
StandardError=append:/opt/orb-live/data/logs/session.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable orb-live
sudo systemctl start orb-live
sudo systemctl status orb-live
```

---

## 5. Cron Jobs

Install for the `orb` user (`crontab -e`):

```cron
# Daily archive rollup (16:30 ET = 21:30 UTC, Mon–Fri)
30 21 * * 1-5  cd /opt/orb-live && .venv/bin/python -m orb_live.ops.long_term_logging

# Nightly backup (22:00 UTC)
0 22 * * 1-5   cd /opt/orb-live && source .env && .venv/bin/python -m orb_live.scripts.backup_state_store

# Log cleanup (daily, 23:00 UTC)
0 23 * * *     cd /opt/orb-live && .venv/bin/python -c "from orb_live.ops.long_term_logging import cleanup_old_logs; from pathlib import Path; cleanup_old_logs(Path('data/logs'))"

# Quarterly calibration (Jan/Apr/Jul/Oct 1, 12:00 UTC)
0 12 1 1,4,7,10 *  cd /opt/orb-live && source .env && .venv/bin/python -m orb_live.ops.quarterly_calibration full
```

---

## 6. Monitoring Setup

### Prometheus + Grafana (optional)

The `/metrics` endpoint serves Prometheus-compatible text at `http://localhost:8080/metrics`.

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: orb_live
    static_configs:
      - targets: ["localhost:8080"]
    scrape_interval: 60s
```

Key metrics:
| Metric | Type | Description |
|--------|------|-------------|
| `orb_bars_received_total` | counter | Bars received per symbol |
| `orb_open_positions_gauge` | gauge | Current open positions |
| `orb_today_pnl_gauge` | gauge | Today's realized P&L ($) |
| `orb_account_equity_gauge` | gauge | Account equity ($) |
| `orb_ws_reconnects_total` | counter | Total WebSocket reconnects |
| `orb_last_reconnect_age_seconds_gauge` | gauge | Seconds since last reconnect |

### Simple uptime check (without Grafana)

Add to crontab:
```cron
# Health check every 5 minutes; alert via email if 503
*/5 * * * *  curl -sf http://localhost:8080/health > /dev/null || echo "ORB UNHEALTHY $(date)" | mail -s "ORB Alert" you@example.com
```

### Firewall

The health server binds to `127.0.0.1:8080` — not exposed externally.
Only allow external access if behind a reverse proxy with authentication.

```bash
# Allow only localhost connections to 8080 (default — already safe)
sudo ufw allow 22/tcp      # SSH
sudo ufw enable
```
