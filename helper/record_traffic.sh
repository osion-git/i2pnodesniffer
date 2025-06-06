#!/bin/bash

# Starte tcpdump im Hintergrund
sudo tcpdump -nnni any net "10.8.0.0/16" -w traffic3.pcap &
TCPDUMP_PID=$!
sleep 2
# Starte die Anfrage-Schleife
for i in {0..1000}; do
  echo "==================== Anfrage $i von 1000 ====================" 
  echo ""
  curl --proxy http://192.168.122.227:4444 \
       http://hpm3retsmpjawh2l7ounhdnngdxcwpxu44wy5z5nu32f2qpjim6q.b32.i2p/  
  echo "" 
  sudo -v
done

# Sende Ctrl+C (SIGINT) an tcpdump
sudo kill -SIGINT "$TCPDUMP_PID"
echo "tcpdump wurde korrekt mit SIGINT beendet."
