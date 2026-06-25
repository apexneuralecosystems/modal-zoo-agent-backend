# Agent Setup & Publish

## Publish a version
Run inside: `Combined-Vision/modal-zoo-agent-backend`
Local:
```
cd D:\ApexNeural\Vision\v\Combined-Vision\modal-zoo-agent-backend
python publish_release.py 1.0.1 --server http://localhost:3001 --email admin@apexneural.com --password ChangeMe@123 --notes "msg"
```
Cloud: same command, change `--server` to your cloud URL.

## Download zip
Dashboard → Admin → Agent → Agent Versions → Download zip.

## Test on Windows
Run inside: `Combined-Vision/modal-zoo-agent-backend`
```
cd D:\ApexNeural\Vision\v\Combined-Vision\modal-zoo-agent-backend
pip install -r requirements.txt
python local_test.py D:\agent-run
```
Edit `D:\agent-run\config.json` → set `server_url`, `secret_token`, `branch_id`, `mac_serial` → re-run `python local_test.py D:\agent-run`.

## Install on Mac
1. Unzip the downloaded zip (normal double-click is fine).
2. Open Terminal and go INTO the unzipped folder (where all the files are):
```
cd /path/to/unzipped-agent-folder
```
3. Run these, in order:
```
bash setup.sh
nano config.json      # set server_url, secret_token, branch_id, mac_serial — save & close
launchctl load ~/Library/LaunchAgents/com.apexneural.visionagent.plist
```

## Update running Macs
Dashboard → Mac Fleet → select Macs → Update to Latest / Stable.
