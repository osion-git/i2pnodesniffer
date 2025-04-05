import subprocess
import requests
import re

# Extrahiert Routerdaten aus Kubernetes, ruft anschliessend die I2P-Webkonsole auf und extrahiert weitere Routerinformationen.
# Am Ende werden die Daten in einer Datei gespeichert. Es ist damit möglich jeder Knoten einer IP zuzuweisen.
def get_ports():
    print("Ports über 'kubectl' Abfragen")
    result = subprocess.run(['kubectl', 'get', 'svc', '-n', 'i2pd'], capture_output=True, text=True)
    output = result.stdout
    print(f"Debug - Kubectl Output:\n{output}")

    ports = []
    lines = output.strip().splitlines()
    # Überspringe Header
    for line in lines[1:]:
        columns = line.split()
        if len(columns) >= 5:
            port_column = columns[4]
            if '7070:' in port_column:
                node_port = port_column.split(':')[1].split('/')[0]
                ports.append(node_port)

    print("Gefundene Ports für die Nodes:")
    for port in ports:
        print(f" - {port}")
    return ports

def fetch_router_info(url):
    print(f"\nVersuche die Routerinformationen von {url} abzurufen...")
    try:
        response = requests.get(url)
        print(f"Debug - HTTP-Statuscode: {response.status_code}")
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Fehler beim Abrufen von {url}: {e}")
        return None

def extract_data(html):
    print("Extrahiere Routerinformationen...")
    router_ident = re.search(r'<b>Router Ident:</b>\s*([^<]+)<br>', html)
    router_caps = re.search(r'<b>Router Caps:</b>\s*([^<]+)<br>', html)
    version = re.search(r'<b>Version:</b>\s*([^<]+)<br>', html)
    ntcp2 = re.search(r'<td>NTCP2</td>\s*<td>([^<]+)</td>', html)
    ssu2 = re.search(r'<td>SSU2</td>\s*<td>([^<]+)</td>', html)
    data = {
        "Router Ident": router_ident.group(1) if router_ident else "Unknown",
        "Router Caps": router_caps.group(1) if router_caps else "Unknown",
        "Version": version.group(1) if version else "Unknown",
        "NTCP2": ntcp2.group(1) if ntcp2 else "Unknown",
        "SSU2": ssu2.group(1) if ssu2 else "Unknown",
    }
    print("Extrahierte Daten:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    return data

def write_to_file(data, url):
    print(f"Schreibe Informationen für {url} in die Datei.")
    with open("router_info.txt", "a") as file:
        file.write(f"Router Information für URL: {url}\n")
        file.write("========================================\n")
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
    print(f"Informationen für {url} erfolgreich gespeichert!\n")

def main():
    print("Starte die Router-Info Extraktion.")
    ports = get_ports()
    if not ports:
        print("Keine Ports gefunden. Überprüfe den Dienst.")
        return
    for port in ports:
        url = f"http://127.0.0.1:{port}/"
        html = fetch_router_info(url)
        if html:
            data = extract_data(html)
            write_to_file(data, url)
    print("Alle Routerinformationen wurden gespeichert.")

if __name__ == "__main__":
    main()
