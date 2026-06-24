import csv
import requests
import time

# Dateipfade (Hier anpassen)
input_file = "/beegfs/prj/RNA_NLP/protein_half_lives/Protein_half_lifes.csv"  # Ihre vorhandene CSV-Datei
output_file = "/beegfs/prj/RNA_NLP/protein_half_lives/esm_data/ensembl_gene_mapping.csv" # Die neue Übersetzungs-Datei

# Cache, um doppelte API-Anfragen für dieselbe Gen-ID zu vermeiden
gene_cache = {}

def fetch_gene_symbol(ensg_id):
    """Fragt das offizielle Gen-Symbol über die Ensembl REST API ab."""
    # 1. Prüfen, ob wir dieses Gen schon mal abgefragt haben
    if ensg_id in gene_cache:
        return gene_cache[ensg_id]
    
    # 2. Wenn nicht, API-Anfrage stellen
    url = f"https://rest.ensembl.org/lookup/id/{ensg_id}?content-type=application/json"
    
    try:
        response = requests.get(url)
        
        # Ensembl-spezifisches Rate-Limiting abfangen (Falls wir zu schnell sind)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 1))
            print(f"\n[Info] API-Limit erreicht. Warte {retry_after} Sekunde(n)...")
            time.sleep(retry_after)
            response = requests.get(url) # Erneuter Versuch
            
        if response.ok:
            data = response.json()
            # Der 'display_name' ist das gesuchte Gen-Symbol (z.B. ARF5)
            symbol = data.get("display_name", "Unbekannt")
            
            # Im Cache speichern für die nächsten Zeilen
            gene_cache[ensg_id] = symbol
            return symbol
        else:
            return "Nicht gefunden"
            
    except Exception as e:
        return f"Fehler ({str(e)})"

# ---------------------------------------------------------
# Hauptprogramm: CSV lesen und übersetzte CSV schreiben
# ---------------------------------------------------------
print("Starte die Übersetzung der Ensembl-IDs...")

try:
    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        
        # Reader und Writer aufsetzen
        reader = csv.DictReader(infile)
        
        # Wir behalten Ihre alten Spalten und fügen die neue "gene_symbol" hinzu
        fieldnames = reader.fieldnames + ["gene_symbol"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Header in die neue Datei schreiben
        writer.writeheader()
        
        zeilen_zaehler = 0
        neu_abfragen = 0
        
        for row in reader:
            zeilen_zaehler += 1
            ensg_id = row.get("gene")
            
            if ensg_id:
                # Prüfen, ob es eine neue Abfrage ist (nur für die Statistik)
                if ensg_id not in gene_cache:
                    neu_abfragen += 1
                
                # Gen-Symbol holen (aus API oder Cache)
                symbol = fetch_gene_symbol(ensg_id)
                row["gene_symbol"] = symbol
            else:
                row["gene_symbol"] = "Keine Gen-ID"
            
            # Zeile in die neue Datei schreiben
            writer.writerow(row)
            
            # Fortschrittsanzeige in der Konsole
            if zeilen_zaehler % 10 == 0:
                print(f"Verarbeitet: {zeilen_zaehler} Zeilen... (API-Abfragen bisher: {neu_abfragen})", end="\r")
                
except FileNotFoundError:
    print(f"\n[Fehler] Die Datei '{input_file}' wurde nicht gefunden. Bitte Dateipfad prüfen.")
    exit()

print(f"\n\nFertig! {zeilen_zaehler} Zeilen wurden verarbeitet.")
print(f"Es mussten nur {neu_abfragen} eindeutige Gene von Ensembl geladen werden.")
print(f"Die Ergebnisse wurden in '{output_file}' gespeichert.")