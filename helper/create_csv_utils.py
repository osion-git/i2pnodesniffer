import os

import pandas as pd
from scapy.all import IP, TCP, UDP, Raw


def create_csv(packets, filename='packets_export.csv', append=False):
    """
    Exports a list of Scapy packages to a CSV file.

    :param packets: List of Scapy packages
    :param filename: File name for the CSV file
    :param append: If True, appends the packets to a CSV file
    """
    data = []

    for frame_num, pkt in enumerate(packets):
        row = {
            'frame_num': frame_num,
            'old_frame_num': pkt.old_frame_number,
            'timestamp': pkt.time,
            # 'src_ip': int(ipaddress.IPv4Address(pkt[IP].src)) if IP in pkt else 0,
            # 'dst_ip': int(ipaddress.IPv4Address(pkt[IP].dst)) if IP in pkt else 0,
            # 'ip_ihl': pkt[IP].ihl if IP in pkt else -1,
            # 'ip_tos': pkt[IP].tos if IP in pkt else -1,
            # 'ip_len': pkt[IP].len if IP in pkt else -1,
            # 'ip_id': pkt[IP].id if IP in pkt else -1,
            # 'ip_flags': pkt[IP].flags.value if IP in pkt else 0,
            # 'ip_frag': pkt[IP].frag if IP in pkt else -1,
            # 'ip_ttl': pkt[IP].ttl if IP in pkt else -1,
            # 'ip_proto': pkt[IP].proto if IP in pkt else -1,
            # TCP fields
            # 'src_port': pkt[TCP].sport if TCP in pkt else -1,
            # 'dst_port': pkt[TCP].dport if TCP in pkt else -1,
            # 'tcp_seq': pkt[TCP].seq if TCP in pkt else -1,
            # 'tcp_ack': pkt[TCP].ack if TCP in pkt else -1,
            # 'tcp_dataofs': pkt[TCP].dataofs if TCP in pkt else -1,
            # 'tcp_flags': pkt[TCP].flags.value if TCP in pkt else 0,
            # 'tcp_window': pkt[TCP].window if TCP in pkt else -1,
            # 'tcp_urgptr': pkt[TCP].urgptr if TCP in pkt else -1,
            # UDP fields
            # 'udp_sport': pkt[UDP].sport if UDP in pkt else -1,
            # 'udp_dport': pkt[UDP].dport if UDP in pkt else -1,
            # 'udp_len': pkt[UDP].len if UDP in pkt else -1,
            # Payload
            # 'payload_size': len(pkt[Raw].load) if Raw in pkt else 0,
            'payload_bytes': ','.join(str(b) for b in pkt[Raw].load) if Raw in pkt else '',
            'prediction': pkt.prediction
        }
        data.append(row)

    df = pd.DataFrame(data)
    if append and os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"Data appended to existing CSV file '{filename}'.")
    else:
        df.to_csv(filename, index=False)
        print(f"CSV file '{filename}' has been created.")
