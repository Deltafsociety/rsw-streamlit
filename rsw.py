import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import csv
from io import StringIO, BytesIO
import zipfile
import xml.etree.ElementTree as ET
import datetime
import sys
import time
from urllib.parse import urljoin
import json

# --- Configuration ---
MAX_REPORT_AGE_SECONDS = 3600 * 24 # 24 hours (adjust as needed)
OFAC_SDN_CSV_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
UK_SANCTIONS_PUBLICATION_PAGE_URL = "https://www.gov.uk/government/publications/the-uk-sanctions-list"
DMA_XLSX_URL = "https://www.dma.dk/Media/638834044135010725/2025118019-7%20Importversion%20-%20List%20of%20EU%20designated%20vessels%20(20-05-2025)%203010691_2_0.XLSX"
UANI_WEBSCRAPE_URL = "https://www.unitedagainstnucleariran.com/blog/switch-list-tankers-shift-from-carrying-iranian-oil-to-russian-oil"
UANI_BUNDLED_CSV_NAME = "UANI_Switch_List_Bundled.csv" # Name of the CSV file to bundle for UANI
MY_VESSELS_CSV_NAME = "my_vessels.csv"
USER_UPLOADED_SANCTIONS_CACHE_NAME = "user_uploaded_sanctions_cache.csv" # For general user uploads

# Define common column patterns for numerical identifiers like IMO numbers
IMO_LIKE_COLUMN_PATTERNS = ['imo', 'id', 'number', 'code', 'vesselimo', 'imo no']

# --- Global Session for Network Requests (cached) ---
@st.cache_resource
def get_requests_session():
    """Configures a requests session with retry strategy and a User-Agent."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from fake_useragent import UserAgent

    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Try to get a random User-Agent, fall back to static if fake_useragent fails
    try:
        ua = UserAgent()
        user_agent_string = ua.random
    except Exception:
        user_agent_string = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'

    session.headers.update({'User-Agent': user_agent_string})
    return session

# --- Helper: Get Soup from URL (removed st.info, st.success, st.error) ---
def get_soup_silent(url, is_xml=False, delay_seconds=1):
    time.sleep(delay_seconds) # Respect delays
    session = get_requests_session()
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        parser_type = 'lxml-xml' if is_xml else 'lxml'
        soup = BeautifulSoup(response.content, parser_type)
        return soup
    except requests.exceptions.RequestException as e:
        # In a cached function, avoid direct Streamlit calls here.
        # Log to console or internal variable if critical for debugging, but not GUI.
        return None
    except Exception as e:
        return None

# --- Data Fetching Functions ---

@st.cache_data(ttl=MAX_REPORT_AGE_SECONDS)
def fetch_ofac_vessels():
    vessels_data = []
    try:
        session = get_requests_session()
        response = session.get(OFAC_SDN_CSV_URL, timeout=60)
        response.raise_for_status()
        csv_content = response.content.decode('utf-8')
        
        reader = csv.reader(StringIO(csv_content))
        for row_idx, row in enumerate(reader):
            if len(row) > 2 and row[2].strip().upper() == 'VESSEL':
                name = row[1].strip()
                imo_number = "N/A"
                if len(row) > 11 and row[11]:
                    remarks = row[11].strip().upper()
                    imo_match = re.search(r'IMO\s*(\d{7})', remarks)
                    if imo_match:
                        imo_number = imo_match.group(1)
                        imo_number = re.sub(r'\D', '', imo_number).strip()
                if imo_number != "N/A" and re.fullmatch(r'^\d{7}$', imo_number):
                    vessels_data.append({"Vessel Name": name, "IMO Number": imo_number, "Source": "OFAC"})
        
        df_vessels = pd.DataFrame(vessels_data)
        if not df_vessels.empty:
            df_vessels.drop_duplicates(subset=['IMO Number'], keep='first', inplace=True)
        return df_vessels
    except requests.exceptions.RequestException as e:
        st.session_state.logs.append(f"ERROR: OFAC Network error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.session_state.logs.append(f"ERROR: OFAC Processing error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=MAX_REPORT_AGE_SECONDS)
def fetch_uk_sanctions_vessels():
    vessels_data = []
    try:
        main_page_soup = get_soup_silent(UK_SANCTIONS_PUBLICATION_PAGE_URL)
        if not main_page_soup:
            st.session_state.logs.append("ERROR: UK: Failed to get main page soup.")
            return pd.DataFrame()
        
        odt_link_tag = main_page_soup.find('a', href=re.compile(r'\.odt$', re.IGNORECASE))
        if not odt_link_tag:
            st.session_state.logs.append("ERROR: UK: Could not find ODT download link on UK Sanctions page.")
            return pd.DataFrame()
        
        odt_url = urljoin(UK_SANCTIONS_PUBLICATION_PAGE_URL, odt_link_tag['href'])
        
        session = get_requests_session()
        odt_response = session.get(odt_url, timeout=60)
        odt_response.raise_for_status()
        
        vessels_data = []
        try:
            with zipfile.ZipFile(BytesIO(odt_response.content)) as odt_zip:
                content_xml_name = 'content.xml'
                if content_xml_name not in odt_zip.namelist():
                    xml_files = [name for name in odt_zip.namelist() if name.endswith('.xml') and 'content' in name.lower()]
                    if xml_files:
                        content_xml_name = xml_files[0]
                    else:
                        st.session_state.logs.append("ERROR: UK: No suitable content XML file found in ODT.")
                        return pd.DataFrame()

                content_xml = odt_zip.read(content_xml_name)
                
                root = ET.fromstring(content_xml)
                full_text = "".join(root.itertext()).strip()

                imo_matches = list(re.finditer(r'IMO\s*[:;]?\s*(\d{7})', full_text, re.IGNORECASE))

                for match in imo_matches:
                    imo = match.group(1)
                    context_start = max(0, match.start() - 200)
                    context = full_text[context_start : match.start()]
                    
                    name_pattern = r'(?:Name\s*[:;]\s*|Vessel Name\s*[:;]\s*)([^\n,仿佛;]{5,100}?)\s*vessel' # Updated regex
                    name_match = re.search(name_pattern, context, re.IGNORECASE)
                    
                    name = "Unknown UK Vessel"
                    if name_match:
                        name = name_match.group(1).strip()
                        name = re.sub(r'^(?:Name|Vessel Name)\s*[:;]?\s*', '', name, flags=re.IGNORECASE).strip()
                        if len(name) > 100: name = name[:100] + "..."
                    elif "vessel" in context.lower():
                        generic_name_match = re.search(r'([^\n,;]{5,100}?)\s+vessel', context, re.IGNORECASE)
                        if generic_name_match:
                            name = generic_name_match.group(1).strip()
                            if len(name) > 100: name = name[:100] + "..."

                    vessels_data.append({"Vessel Name": name, "IMO Number": imo, "Source": "UK (ODT)"})
            
            df_vessels = pd.DataFrame(vessels_data)
            return df_vessels
        except zipfile.BadZipFile:
            st.session_state.logs.append("ERROR: UK: Downloaded UK ODT file is not a valid zip file. It might be corrupted or not an ODT.")
            return pd.DataFrame()
        except ET.ParseError as e:
            st.session_state.logs.append(f"ERROR: UK: XML parsing error for ODT content: {e}. The ODT file might be malformed or its internal XML structure changed.")
            return pd.DataFrame()
        except Exception as e: # Catch any other unexpected errors during ODT processing
            st.session_state.logs.append(f"ERROR: UK: An unexpected error occurred during UK Sanctions processing: {e}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.session_state.logs.append(f"ERROR: UK Network error during ODT download: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.session_state.logs.append(f"ERROR: UK General error before ODT parsing: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=MAX_REPORT_AGE_SECONDS)
def fetch_dma_vessels():
    vessels_data = []
    try:
        session = get_requests_session()
        response = session.get(DMA_XLSX_URL, timeout=60)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))
        
        imo_col = None
        name_col = None
        for col in df.columns:
            cleaned_col = str(col).lower().strip()
            if 'imo' in cleaned_col and not imo_col:
                imo_col = col
            if ('vessel' in cleaned_col or 'name' in cleaned_col) and not name_col:
                name_col = col
            if imo_col and name_col: break

        if not imo_col or not name_col:
            st.session_state.logs.append(f"ERROR: DMA: Could not find 'IMO' or 'Vessel Name' columns. Found: {df.columns.tolist()}.")
            return pd.DataFrame()
        
        df = df[[name_col, imo_col]].copy()
        df.rename(columns={name_col: "Vessel Name", imo_col: "IMO Number"}, inplace=True)
        df["Source"] = "EU (DMA)"
        df.dropna(subset=["IMO Number"], inplace=True)
        df["IMO Number"] = df["IMO Number"].astype(str).apply(lambda x: re.sub(r'\D', '', str(x)))
        df = df[df['IMO Number'].astype(str).str.contains(r'^\d{7}$')]
        
        return df
    except requests.exceptions.RequestException as e:
        st.session_state.logs.append(f"ERROR: DMA Network error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.session_state.logs.append(f"ERROR: DMA Processing error: {e}")
        return pd.DataFrame()

# UANI Fetching with bundled CSV priority and webscrape fallback (removed UI calls)
@st.cache_data(ttl=MAX_REPORT_AGE_SECONDS)
def fetch_uani_vessels():
    df_from_bundled = pd.DataFrame()
    
    # Try to load from bundled CSV first
    try:
        current_dir = os.path.dirname(__file__)
        bundled_csv_full_path = os.path.join(current_dir, UANI_BUNDLED_CSV_NAME)
        
        if os.path.exists(bundled_csv_full_path):
            st.session_state.logs.append(f"INFO: UANI: Attempting to load from bundled CSV: {UANI_BUNDLED_CSV_NAME}")
            df = pd.read_csv(bundled_csv_full_path)
            
            original_columns = list(df.columns)
            cleaned_columns_map = {str(col).strip().lower(): col for col in original_columns}
            
            imo_col_key = None
            name_col_key = None

            for pattern in ['imo', 'imo number', 'imo no']:
                if pattern in cleaned_columns_map:
                    imo_col_key = cleaned_columns_map[pattern]
                    break
            
            for pattern in ['vessel name', 'vessel', 'name']:
                if pattern in cleaned_columns_map:
                    name_col_key = cleaned_columns_map[pattern]
                    break
            
            if imo_col_key and name_col_key:
                df_from_bundled = df[[name_col_key, imo_col_key]].copy()
                df_from_bundled.rename(columns={name_col_key: "Vessel Name", imo_col_key: "IMO Number"}, inplace=True)
                df_from_bundled["Source"] = "UANI (Bundled CSV)"
                df_from_bundled.dropna(subset=["IMO Number"], inplace=True)
                df_from_bundled["IMO Number"] = df_from_bundled["IMO Number"].astype(str).apply(lambda x: re.sub(r'\D', '', str(x)))
                df_from_bundled = df_from_bundled[df_from_bundled['IMO Number'].str.match(r'^\d{7}$')]
                st.session_state.logs.append(f"SUCCESS: UANI: Loaded {len(df_from_bundled)} vessels from bundled CSV.")
                return df_from_bundled # Return immediately if successful
            else:
                st.session_state.logs.append(f"WARNING: UANI: Bundled CSV headers missing 'IMO' or 'Vessel Name'. Headers: {original_columns}. Falling back to web scrape.")

    except Exception as e:
        st.session_state.logs.append(f"WARNING: UANI: Error loading bundled CSV ({UANI_BUNDLED_CSV_NAME}): {e}. Falling back to web scrape.")

    # Fallback to web scraping if bundled CSV fails or doesn't exist/is malformed
    st.session_state.logs.append(f"INFO: UANI: Attempting web scrape from {UANI_WEBSCRAPE_URL} (this may be blocked)...")
    try:
        soup = get_soup_silent(UANI_WEBSCRAPE_URL)
        if not soup:
            st.session_state.logs.append("ERROR: UANI: Failed to get soup from UANI URL (web scrape fallback).")
            return pd.DataFrame()

        st.session_state.logs.append("INFO: UANI: HTML content downloaded. Searching for tables...")
        
        tables = soup.find_all('table')
        vessels_data = []
        
        for table_idx, table in enumerate(tables):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            
            imo_col_idx = -1
            name_col_idx = -1
            
            normalized_headers = [h.lower().replace('*', '').strip() for h in headers]
            
            for idx, header in enumerate(normalized_headers):
                if 'imo' in header and imo_col_idx == -1:
                    imo_col_idx = idx
                if ('vessel' in header or 'name' in header) and name_col_idx == -1:
                    name_col_idx = idx
            
            if imo_col_idx != -1 and name_col_idx != -1:
                st.session_state.logs.append(f"INFO: UANI: Found relevant table (Table {table_idx+1}) with IMO (col {imo_col_idx}) and Name (col {name_col_idx}) columns.")
                rows = table.find_all('tr')
                
                for r_idx, row in enumerate(rows):
                    if r_idx == 0:
                        continue
                        
                    cols = row.find_all(['td', 'th'])
                    cols = [ele.text.strip() for ele in cols]
                    
                    if len(cols) > max(imo_col_idx, name_col_idx):
                        vessel_name = cols[name_col_idx]
                        imo_number = cols[imo_col_idx]
                        
                        imo_number = re.sub(r'\D', '', str(imo_number)).strip()
                        
                        if re.fullmatch(r'^\d{7}$', imo_number):
                            vessels_data.append({"Vessel Name": vessel_name, "IMO Number": imo_number, "Source": "UANI (Web Scraping)"})
                
                st.session_state.logs.append(f"SUCCESS: UANI: Processed {len(vessels_data)} entries from Table {table_idx+1}.")
                break
            else:
                st.session_state.logs.append(f"INFO: UANI: Table {table_idx+1} does not contain both 'IMO' and 'Vessel Name' columns. Headers: {headers}")

        df = pd.DataFrame(vessels_data)
        
        if not df.empty:
            df.drop_duplicates(subset=['IMO Number'], keep='first', inplace=True)
            
        st.session_state.logs.append(f"SUCCESS: UANI: Final {len(df)} vessels from web scrape.")
        return df
        
    except requests.exceptions.RequestException as e:
        st.session_state.logs.append(f"ERROR: UANI Network error fetching HTML data (web scrape fallback): {e}")
        return pd.DataFrame()
    except Exception as e:
        st.session_state.logs.append(f"ERROR: UANI Processing HTML data (web scrape fallback): {e}")
        return pd.DataFrame()

# --- Main Report Processing Logic ---
def process_all_data():
    st.session_state.logs = [] 
    st.session_state.logs.append("INFO: Starting sanctions report generation...")
    
    fetched_data = {}
    
    progress_bar_placeholder = st.empty() 
    
    def update_overall_progress_value(value, status_text):
        progress_bar_placeholder.progress(value, text=status_text)
        time.sleep(0.01)

    with st.spinner("Fetching OFAC data..."):
        update_overall_progress_value(0.05, "Starting OFAC fetch...")
        fetched_data["OFAC_Vessels"] = fetch_ofac_vessels()
        st.session_state.logs.append(f"INFO: OFAC fetch finished. Found {len(fetched_data['OFAC_Vessels'])} vessels.")
        
    update_overall_progress_value(0.25, "OFAC data fetched. Starting UK fetch...")

    with st.spinner("Fetching UK sanctions data..."):
        fetched_data["UK_Sanctions_Vessels"] = fetch_uk_sanctions_vessels()
        st.session_state.logs.append(f"INFO: UK fetch finished. Found {len(fetched_data['UK_Sanctions_Vessels'])} vessels.")
        
    update_overall_progress_value(0.50, "UK data fetched. Starting EU (DMA) fetch...")

    with st.spinner("Fetching EU (DMA) data..."):
        fetched_data["EU_DMA_Vessels"] = fetch_dma_vessels()
        st.session_state.logs.append(f"INFO: EU (DMA) fetch finished. Found {len(fetched_data['EU_DMA_Vessels'])} vessels.")
        
    update_overall_progress_value(0.75, "EU (DMA) data fetched. Starting UANI fetch...")

    with st.spinner("Fetching UANI data..."):
        fetched_data["UANI_Vessels_Tracked"] = fetch_uani_vessels()
        st.session_state.logs.append(f"INFO: UANI fetch finished. Found {len(fetched_data['UANI_Vessels_Tracked'])} vessels.")
        
    update_overall_progress_value(1.0, "Report generation and data consolidation complete!")
    st.success("Report generation process completed. Check 'Sanctions Report Viewer' tab for data.")
    st.session_state.logs.append("SUCCESS: Report generation process completed.")

    st.session_state.global_sanctions_data_store = fetched_data
    st.session_state.report_generated = True

# --- My Vessels Data Handling Functions ---
def clear_vessel_list():
    """Clears the vessel list in session state."""
    st.session_state.my_vessels_df = pd.DataFrame(columns=['Vessel Name', 'IMO Number'])

def load_vessels_from_uploaded_file(uploaded_file):
    """Loads a CSV file into the session state DataFrame."""
    try:
        df_uploaded_vessels = pd.read_csv(uploaded_file)
        
        name_col = next((col for col in df_uploaded_vessels.columns if 'name' in col.lower().strip()), None)
        imo_col = next((col for col in df_uploaded_vessels.columns if 'imo' in col.lower().strip()), None)

        if name_col and imo_col:
            df_uploaded_vessels = df_uploaded_vessels[[name_col, imo_col]].copy()
            df_uploaded_vessels.rename(columns={name_col: "Vessel Name", imo_col: "IMO Number"}, inplace=True)
            df_uploaded_vessels['IMO Number'] = df_uploaded_vessels['IMO Number'].astype(str).str.replace(r'\D', '', regex=True)
            df_uploaded_vessels = df_uploaded_vessels[df_uploaded_vessels['IMO Number'].str.match(r'^\d{7}$')]
            
            st.session_state.my_vessels_df = df_uploaded_vessels
            st.success(f"Loaded {len(st.session_state.my_vessels_df)} vessels from '{uploaded_file.name}'.")
        else:
            st.error(f"The uploaded CSV must contain columns for 'name' and 'imo'. Found columns: {df_uploaded_vessels.columns.tolist()}")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

# --- Helper for User Uploaded Sanctions Data Persistence ---
def save_user_uploaded_sanctions_to_file(df, file_path):
    """Saves a DataFrame to a specified CSV for user uploads for persistence."""
    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
        st.success(f"User uploaded data saved to {os.path.basename(file_path)} for persistence.")
    except Exception as e:
        st.error(f"Error saving user uploaded data to {os.path.basename(file_path)}: {e}")

def load_user_uploaded_sanctions_from_file(file_path):
    """Loads user uploaded data from a specified CSV for persistence."""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            st.success(f"Loaded user uploaded data from {os.path.basename(file_path)}.")
            return df
        except Exception as e:
            st.warning(f"Could not load previous user uploaded data from {os.path.basename(file_path)}: {e}. Starting fresh.")
    return pd.DataFrame() # Return empty DataFrame if not found or error

# --- Streamlit App Structure ---

st.set_page_config(layout="wide", page_title="Royal Sanction Watch", page_icon=":anchor:")

st.title("⚓ Royal Classification Society: Vessel Sanction & Data Tool")

# Initialize all session state variables at the top
if 'global_sanctions_data_store' not in st.session_state:
    st.session_state.global_sanctions_data_store = {
        "OFAC_Vessels": pd.DataFrame(),
        "UK_Sanctions_Vessels": pd.DataFrame(),
        "EU_DMA_Vessels": pd.DataFrame(),
        "UANI_Vessels_Tracked": pd.DataFrame()
    }
    st.session_state.global_sanctions_data_store["User_Uploaded_Sanctions"] = pd.DataFrame()
if 'my_vessels_data' not in st.session_state:
    st.session_state.my_vessels_data = []
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'logs' not in st.session_state:
    st.session_state.logs = []


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Sanctions Report Generator", "Sanctions Report Viewer", "My Vessels", "About This Application"])

with tab1:
    st.header("Generate Sanctions Report")
    st.markdown("Click the button below to fetch the latest sanctions data from various sources.")
    st.warning("Note: Fetching directly from websites may sometimes encounter anti-bot measures (e.g., 403 Forbidden).")
    
    if st.button("Generate New Sanctions Report"):
        st.session_state.logs = [] 
        process_all_data()
        st.rerun()

    st.subheader("Process Logs")
    for log_message in st.session_state.logs:
        if log_message.startswith("ERROR:"):
            st.error(log_message.replace("ERROR: ", ""))
        elif log_message.startswith("WARNING:"):
            st.warning(log_message.replace("WARNING: ", ""))
        elif log_message.startswith("INFO:"):
            st.info(log_message.replace("INFO: ", ""))
        elif log_message.startswith("SUCCESS:"):
            st.success(log_message.replace("SUCCESS: ", ""))
        else:
            st.write(log_message)


with tab2:
    st.header("Sanctions Report Viewer")
    
    if not any(df is not None and not df.empty for df in st.session_state.global_sanctions_data_store.values()):
        st.info("No sanctions report data available yet. Please generate a report in the 'Sanctions Report Generator' tab.")
    else:
        st.markdown("View the fetched sanctions data here.")

        combined_df = pd.DataFrame()
        sanctioned_imos = set()
        sanctioned_sources = {}

        with st.spinner("Consolidating data for viewer..."):
            all_dfs_raw = [df for df in st.session_state.global_sanctions_data_store.values() if df is not None and not df.empty]
            
            if all_dfs_raw:
                temp_combined_sanctions_df = pd.concat(all_dfs_raw, ignore_index=True)
                for index, row in temp_combined_sanctions_df.iterrows():
                    imo_val_cleaned = re.sub(r'\D', '', str(row['IMO Number'])).strip()
                    if re.fullmatch(r'^\d{7}$', imo_val_cleaned):
                        sanctioned_imos.add(imo_val_cleaned)
                        sanctioned_sources.setdefault(imo_val_cleaned, []).append(row['Source'])
        
        df_display_store = {}
        for source_name, df in st.session_state.global_sanctions_data_store.items():
            if df is not None and not df.empty and 'IMO Number' in df.columns:
                df_copy = df.copy()
                df_copy['Sanctioned?'] = df_copy['IMO Number'].apply(
                    lambda x: 'Yes' if re.sub(r'\D', '', str(x)).strip() in sanctioned_imos else 'No'
                )
                df_copy['Sources'] = df_copy['IMO Number'].apply(
                    lambda x: ', '.join(sanctioned_sources.get(re.sub(r'\D', '', str(x)).strip(), []))
                )
                df_display_store[source_name] = df_copy
            else:
                df_display_store[source_name] = pd.DataFrame()

        all_dfs_processed = [df.assign(Source=source_name) for source_name, df in df_display_store.items() if not df.empty]
        if all_dfs_processed:
            combined_df = pd.concat(all_dfs_processed, ignore_index=True)
            combined_df.drop_duplicates(subset=['IMO Number'], keep='first', inplace=True)
            combined_df = combined_df[['Vessel Name', 'IMO Number', 'Sanctioned?', 'Sources']].sort_values(by='Vessel Name').reset_index(drop=True)
        else:
            combined_df = pd.DataFrame()

        source_options = ["All Data (Combined)"] + [k for k, v in st.session_state.global_sanctions_data_store.items() if v is not None and not v.empty]
        
        selected_source_key = st.selectbox("Select Data Source to View:", source_options)

        df_to_display = pd.DataFrame()
        if selected_source_key == "All Data (Combined)":
            df_to_display = combined_df
        elif selected_source_key in df_display_store:
            df_to_display = df_display_store[selected_source_key]
        
        if not df_to_display.empty:
            st.dataframe(df_to_display, use_container_width=True)

            csv_export = df_to_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Current View as CSV",
                data=csv_export,
                file_name=f"sanctions_view_{selected_source_key.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info(f"No data available for '{selected_source_key}'.")

    st.subheader("Upload External Sanctions Data")
    st.markdown("If fetching fails, you can upload a local CSV or Excel file containing sanctions data. It should have columns for 'IMO' and 'Vessel Name'.")
    uploaded_external_file = st.file_uploader("Upload Sanctions File", type=["csv", "xlsx", "xls"], key="external_sanctions_uploader")
    if uploaded_external_file is not None:
        try:
            if uploaded_external_file.name.endswith('.csv'):
                df_loaded_external = pd.read_csv(uploaded_external_file)
            else:
                df_loaded_external = pd.read_excel(uploaded_external_file)
            
            original_columns = list(df_loaded_external.columns)
            cleaned_columns_map = {str(col).strip().lower(): col for col in original_columns}
            
            imo_col_key = next((col for col in cleaned_columns_map if 'imo' in col), None)
            name_col_key = next((col for col in cleaned_columns_map if 'vessel' in col or 'name' in col), None)

            if imo_col_key and name_col_key:
                df_external_processed = df_loaded_external[[name_col_key, imo_col_key]].copy()
                df_external_processed.rename(columns={name_col_key: "Vessel Name", imo_col_key: "IMO Number"}, inplace=True)
                df_external_processed["Source"] = f"User Uploaded: {uploaded_external_file.name}"
                df_external_processed.dropna(subset=["IMO Number"], inplace=True)
                df_external_processed["IMO Number"] = df_external_processed["IMO Number"].astype(str).apply(lambda x: re.sub(r'\D', '', str(x)))
                df_external_processed = df_external_processed[df_external_processed['IMO Number'].str.match(r'^\d{7}$')]
                
                st.session_state.global_sanctions_data_store["User_Uploaded_Sanctions"] = df_external_processed
                st.success(f"External sanctions data loaded from '{uploaded_external_file.name}'. Refreshing display.")
                st.rerun()
            else:
                st.error(f"Uploaded file '{uploaded_external_file.name}' missing 'IMO' or 'Vessel Name' columns after cleaning. Found headers: {df_loaded_external.columns.tolist()}")
        except Exception as e:
            st.error(f"Error processing uploaded file '{uploaded_external_file.name}': {e}")

    if st.button("Clear All Fetched Sanctions Data"):
        st.session_state.global_sanctions_data_store = {
            "OFAC_Vessels": pd.DataFrame(),
            "UK_Sanctions_Vessels": pd.DataFrame(),
            "EU_DMA_Vessels": pd.DataFrame(),
            "UANI_Vessels_Tracked": pd.DataFrame(),
            "User_Uploaded_Sanctions": pd.DataFrame()
        }
        st.session_state.report_generated = False
        st.success("All fetched sanctions data cleared from memory.")
        st.rerun()

with tab3:
    st.header("My Vessels List")
    st.info("Upload a CSV file or add a vessel manually. The list will be cleared if you close the app.")
    
    if st.button("Clear My Vessel List"):
        st.session_state.my_vessels_data = []
        st.rerun()

    uploaded_my_vessels_file = st.file_uploader("Upload a CSV file with 'Vessel Name' and 'IMO Number' columns", type=["csv"], key="my_vessels_uploader")
    if uploaded_my_vessels_file is not None:
        try:
            with st.spinner("Loading file..."):
                df_uploaded_vessels = pd.read_csv(uploaded_my_vessels_file)
            
            name_col = next((col for col in df_uploaded_vessels.columns if 'name' in col.lower().strip()), None)
            imo_col = next((col for col in df_uploaded_vessels.columns if 'imo' in col.lower().strip()), None)

            if name_col and imo_col:
                df_uploaded_vessels = df_uploaded_vessels[[name_col, imo_col]].copy()
                df_uploaded_vessels.rename(columns={name_col: "Vessel Name", imo_col: "IMO Number"}, inplace=True)
                df_uploaded_vessels['IMO Number'] = df_uploaded_vessels['IMO Number'].astype(str).str.replace(r'\D', '', regex=True)
                df_uploaded_vessels = df_uploaded_vessels[df_uploaded_vessels['IMO Number'].str.match(r'^\d{7}$')]
                
                st.session_state.my_vessels_data = df_uploaded_vessels.to_dict('records')
                st.success(f"Loaded {len(st.session_state.my_vessels_data)} vessels from '{uploaded_my_vessels_file.name}'.")
            else:
                st.error(f"The uploaded CSV must contain columns for 'name' and 'imo'. Found columns: {df_uploaded_vessels.columns.tolist()}")
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
        
        st.rerun()

    st.subheader("Add New Vessel Manually")
    with st.form("add_vessel_form", clear_on_submit=True):
        new_vessel_name = st.text_input("Vessel Name")
        new_vessel_imo = st.text_input("IMO Number (7 digits)")
        add_vessel_submitted = st.form_submit_button("Add Vessel to List")

        if add_vessel_submitted:
            if new_vessel_name and new_vessel_imo:
                if re.fullmatch(r'^\d{7}$', new_vessel_imo):
                    if not any(v['IMO Number'] == new_vessel_imo for v in st.session_state.my_vessels_data):
                        st.session_state.my_vessels_data.append({'Vessel Name': new_vessel_name, 'IMO Number': new_vessel_imo})
                        st.success(f"Added vessel: {new_vessel_name} ({new_vessel_imo})")
                        st.rerun()
                    else:
                        st.warning(f"Vessel with IMO Number {new_vessel_imo} already exists.")
                else:
                    st.error("IMO Number must be exactly 7 digits.")
            else:
                st.error("Both Vessel Name and IMO Number are required.")

    if st.session_state.my_vessels_data:
        my_vessels_df = pd.DataFrame(st.session_state.my_vessels_data)
        
        st.info(f"Displaying {len(my_vessels_df)} vessels from the current list.")
        
        if st.button("Check Sanctions Status & Update List"):
            sanctions_data_available = any(df is not None and not df.empty for df in st.session_state.global_sanctions_data_store.values())
            if not sanctions_data_available:
                st.warning("No fresh sanctions data available. Please generate a report first from the 'Sanctions Report Generator' tab.")
            else:
                sanctions_data_dict = st.session_state.global_sanctions_data_store
                
                sanctioned_imos = set()
                sanctioned_sources = {}
                with st.spinner("Consolidating sanctions lists..."):
                    all_sanctions_df_raw = [df for df in sanctions_data_dict.values() if df is not None and not df.empty]
                    if all_sanctions_df_raw:
                        temp_combined_sanctions_df = pd.concat(all_sanctions_df_raw, ignore_index=True)
                        for index, row in temp_combined_sanctions_df.iterrows():
                            imo_val_cleaned = re.sub(r'\D', '', str(row['IMO Number'])).strip()
                            if re.fullmatch(r'^\d{7}$', imo_val_cleaned):
                                sanctioned_imos.add(imo_val_cleaned)
                                sanctioned_sources.setdefault(imo_val_cleaned, []).append(row['Source'])
                st.info(f"Consolidated {len(sanctioned_imos)} unique sanctioned IMOs.")
                
                my_vessels_df['Sanctioned?'] = my_vessels_df['IMO Number'].astype(str).apply(
                    lambda x: 'Yes' if x in sanctioned_imos else 'No'
                )
                my_vessels_df['Sources'] = my_vessels_df['IMO Number'].astype(str).apply(
                    lambda x: ', '.join(sorted(list(set(sanctioned_sources.get(x, [])))))
                )
                
                st.session_state.my_vessels_data = my_vessels_df.to_dict('records')
                st.success("Sanction check complete. Results are shown in the table below.")
                st.rerun()

        if not my_vessels_df.empty:
            def highlight_sanctioned(row):
                if 'Sanctioned?' in row and row['Sanctioned?'] == 'Yes':
                    return ['background-color: #FFCCCC; color: #CC0000; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            styled_df = my_vessels_df.style.apply(highlight_sanctioned, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
            
            csv_export_my_vessels = my_vessels_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export My Vessels List as CSV",
                data=csv_export_my_vessels,
                file_name=f"my_vessels_list_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="export_my_vessels_btn"
            )
    else:
        st.info("Please upload a CSV file or add a vessel manually to see your list here.")

with tab4:
    st.header("About This Application")
    st.markdown("""
    This application serves as a comprehensive tool designed for the maritime industry, 
    specifically for compliance and risk assessment. It automates the process of exploring 
    the internet to fetch and consolidate critical vessel sanctions data from various 
    authoritative sources, and provides robust tools for local data comparison and management.

    ### How to Use This Program:

    **1. Sanctions Report Generator Tab:**
    * Click 'Generate New Sanctions Report'.
    * The application will then connect to various online sources (OFAC, UK, EU DMA, UANI Website) 
        to fetch the latest vessel sanctions data.
    * Animated progress and status messages will be displayed on this tab.
    * Once complete, the fetched data will be automatically available in the 'Sanctions Report Viewer' tab.

    **2. Sanctions Report Viewer Tab:**
    * This tab automatically displays the data fetched by the 'Sanctions Report Generator'.
    * Use the 'Select Data Source' dropdown to view individual lists (OFAC, UK, EU DMA, UANI) or a combined list.
    * Click 'Download Current View as CSV' to save the currently displayed table to a CSV file.
    * **Upload External Sanctions Data:** You can upload any CSV or Excel file containing sanctions data (ensure "IMO" and "Vessel Name" columns exist). This will add the data to the current session's sanctions store.
    * Click 'Clear All Fetched Sanctions Data' to remove all fetched data from memory and the local cache.

    **3. My Vessels Tab:**
    * **Upload:** Use the 'Upload a CSV file' button to select your vessel list. The app will immediately process it and display it in a table below.
    * **Manual Entry:** Use the form to manually add a new vessel one by one.
    * **Check:** Click 'Check Sanctions Status & Update List' to compare your vessels against the latest sanctions data. The result will be added as new columns in the table.
    * **Export:** Click 'Export My Vessels List as CSV' to download the final list, including the sanctions status.
    * **Note:** The list is **not saved** between sessions. You must re-upload or manually re-enter your data each time you use the app.

    ### Contact Information:
    For technical assistance or inquiries, please contact: `it@rcsclass.org`
    """)