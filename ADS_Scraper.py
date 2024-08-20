import keyring, requests, pandas as pd, time, os, re, ssl
from fuzzywuzzy import process, fuzz
import plotly.express as px, plotly.graph_objects as go
from html import unescape
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox
import json

# keyring.set_password("ADS_API_TOKEN", "ADS_API_TOKEN", "ENTER API TOKEN HERE")
token = keyring.get_password("ADS_API_TOKEN", "ADS_API_TOKEN")
ssl_context = ssl._create_unverified_context()

learned_mapping = {}
carnegie_df = None

user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
nasa_ads_scraper_dir = os.path.join(user_desktop, "CubeSat Data")
data_dir = os.path.join(nasa_ads_scraper_dir, "Data")
merger_dir = os.path.join(nasa_ads_scraper_dir, "Merger")
input_dir = os.path.join(merger_dir, "Input")
output_dir = os.path.join(merger_dir, "Output")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

settings_file = os.path.join(nasa_ads_scraper_dir, ".settings.json")

def load_settings():
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(settings_file, 'w') as f:
        json.dump(settings, f)

settings = load_settings()

def clean_affiliation(affiliation):
    affiliation = unescape(affiliation)
    affiliation = re.sub(r'\buniv\b', 'university', affiliation, flags=re.IGNORECASE)
    affiliation = re.sub(r'\b(united states|usa|united states of america|us)\b', '', affiliation, flags=re.IGNORECASE)
    affiliation = re.sub(r'-', ' ', affiliation)
    return re.sub(r'[\W_]+', ' ', affiliation).strip().lower()


def map_nasa_center(affiliation):
    affiliation = affiliation.lower()
    if 'nasa' in affiliation and 'langley' in affiliation:
        return 'NASA Langley Research Center'
    if any(term in affiliation for term in ['jet propulsion laboratory', 'jpl', 'jet propulsion lab']):
        return 'NASA Jet Propulsion Laboratory'
    if 'nasa' in affiliation and any(term in affiliation for term in ['msfc', 'marshall', 'marshal']):
        return 'NASA Marshall Space Flight Center'
    if any(term in affiliation for term in ['greenbelt', 'goddard', 'gsfc', 'creenbelt']):
        return 'NASA Goddard Space Flight Center'
    if any(term in affiliation for term in ['wff', 'wallops']):
        return 'NASA Wallops Flight Facility'
    if any(term in affiliation for term in ['ames', 'moffett']):
        return 'NASA Ames Research Center'
    if any(term in affiliation for term in ['hq', 'headquarters']):
        return 'NASA HQ'
    if any(term in affiliation for term in ['glenn', 'cleveland', 'oh']):
        return 'NASA Glenn Research Center'
    if any(term in affiliation for term in ['johnson', 'houston']):
        return 'NASA Johnson Space Center'
    if any(term in affiliation for term in ['lunar', 'boulder']):
        return 'NASA Lunar Science Center'
    
    return affiliation.upper() if 'nasa' in affiliation else affiliation

def map_affiliation(affiliation, carnegie_df):
    cleaned_affiliation = clean_affiliation(affiliation)
    if cleaned_affiliation in learned_mapping:
        return learned_mapping[cleaned_affiliation]
    institution_names = [clean_affiliation(name) for name in carnegie_df['Institution'].tolist()]
    match = process.extractOne(cleaned_affiliation, institution_names, scorer=fuzz.token_sort_ratio, score_cutoff=60)
    if match:
        best_match = match[0]
        learned_mapping[cleaned_affiliation] = best_match
        return best_match
    return affiliation

import re
import pandas as pd

def get_carnegie_classification(cleaned_affiliation, carnegie_df):
    cleaned_affiliation = cleaned_affiliation.lower()
    
    if re.search(r'nasa|goddard|gsfc|langley|marshall|msfc|greenbelt|goddard|wallops|wff|ames|moffett|hq|headquarters|glenn|cleveland|johnson|houston|lunar|boulder', cleaned_affiliation, re.IGNORECASE):
        return "NASA Center"
    
    international_keywords = [
        r'italy', r'belgium', r'germany', r'uk', r'united kingdom', r'luxembourg',
        r'france', r'paris', r'israel', r'switzerland', r'netherlands', r'canada',
        r'australia', r'brazil', r'poland', r'spain', r'japan', r'india',
        r'saudi arabia', r'united arab emirates', r'austria', r'sweden', r'norway',
        r'denmark', r'ireland', r'finland', r'iceland', r'argentina', r'chile', r'peru',
        r'uruguay', r'venezuela', r'colombia', r'ecuador', r'russia', r'china', r'korea',
        r'hungary', r'tübingen', r'bulgaria', r'romania', r'slovakia', r'czech republic', r'puerto rico',
        r'new zealand', r'south africa', r'south korea', r'taiwan', r'indonesia', r'philippines',
        r'malaysia', r'vietnam', r'pakistan', r'iran', r'egypt', r'czech', r'singapore'
        r'russian', r'russian federation', r'british columbia canada', r'republic of china'                              
    ]
    
    if any(re.search(keyword, cleaned_affiliation, re.IGNORECASE) for keyword in international_keywords) and not re.search(r'new mexico', cleaned_affiliation, re.IGNORECASE):  
        return "International"
    
    special_cases = {
        r'pennsylvania state university|penn state': "pennsylvania state university",
        r'university of michigan|michigan': "university of michigan",
        r'university of iowa department of physics and astronomy iowa city iowa|department of physics and astronomy university of iowa iowa city ia 52242|u iowa iowa city ia': "university of iowa",
        r'southwest research institute colorado|co': "university of colorado",
        r'colorado': "university of colorado",
        r'embry riddle daytona': "Embry-Riddle Aeronautical University-Daytona Beach",
        r'university texas dallas|tx': "Concordia University Texas",
        r'university of texas san antonio': "The University of Texas at San Antonio",
        r'university of texas dallas': "The University of Texas at Dallas",
        r'university of texas austin': "The University of Texas at Austin",
        r'university of texas houston': "The University of Texas at Houston",
        r'caltech|jet propulsion|jpl|pasadena|california institute of technology': "California Institute of Technology",
        r'berkeley': "University of California, Berkeley",
        r'university of california los angeles': "University of California, Los Angeles",
        r'university of california san diego': "University of California, San Diego",
        r'university of california san francisco': "University of California, San Francisco",
        r'university of california santa barbara': "University of California, Santa Barbara",
        r'harvard': "Harvard University",
        r'columbia': "Columbia University",
        r'boston university': "Boston University",
        r'hopkins': "Johns Hopkins University",
        r'stanford': "Stanford University",
        r'mit': "Massachusetts Institute of Technology",
        r'princeton': "Princeton University",
        r'yale': "Yale University",
        r'university of chicago': "University of Chicago",
        r'university of north carolina': "University of North Carolina",
        r'university of pennsylvania': "University of Pennsylvania",
        r'cambridge': "Cambridge College",
        r'duke': "Duke University",
        r'utah state': "Utah State University",
        r'morehead': "Morehead State University",
        r'noaa': "NOAA",
        r'new mexico': "University of New Mexico"
        
    }
    
    if re.search(r'research institute|research institution|research', cleaned_affiliation, re.IGNORECASE):
        return "Research Institution"
    
    for pattern, institution in special_cases.items():
        if re.search(pattern, cleaned_affiliation, re.IGNORECASE):
            result = carnegie_df[carnegie_df['Institution'].str.contains(institution, case=False)]
            if not result.empty:
                return result['Carnegie Classification'].values[0]
    
    if cleaned_affiliation in carnegie_df['Institution'].str.lower().tolist():
        carnegie_entry = carnegie_df[carnegie_df['Institution'].str.lower() == cleaned_affiliation]
        if not carnegie_entry.empty:
            return carnegie_entry['Carnegie Classification'].values[0]

    return "Unknown"

def fetch_article_data(bibcode, headers):
    r = requests.get(f"https://api.adsabs.harvard.edu/v1/search/query?q=bibcode:{bibcode}&fl=title,abstract,year,author,aff,citation,property,citation_count,doctype", headers=headers)
    if r.status_code == 200:
        return r.json().get("response", {}).get("docs", [])
    return []

def fetch_titles(codes, headers):
    titles = []
    for code in codes:
        cr = requests.get(f"https://api.adsabs.harvard.edu/v1/search/query?q=bibcode:{code}&fl=title", headers=headers)
        if cr.status_code == 200 and cr.json().get("response", {}).get("docs"):
            title = cr.json()["response"]["docs"][0].get("title", ["N/A"])[0]
            titles.append(title)
    return titles

def get_article_info(bibcodes, token, carnegie_df, skip_no_abstract=False):
    articles, articles_separated, carnegie_classifications = [], [], []
    headers = {"Authorization": "Bearer " + token}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_bibcode = {executor.submit(fetch_article_data, bibcode, headers): bibcode for bibcode in bibcodes}
        
        for future in as_completed(future_to_bibcode):
            docs = future.result()
            if not docs:
                continue
            
            doc = docs[0]
            if skip_no_abstract and 'abstract' not in doc:
                continue 

            title = doc.get("title", ["N/A"])[0]
            abstract = doc.get("abstract", "N/A")
            year = doc.get("year", "")
            authors = doc.get("author", [])
            affiliations = doc.get("aff", [])
            
            if not affiliations:
                affiliations = ["N/A"] * len(authors)
                
            citations = doc.get("citation", [])
            citation_count = doc.get("citation_count", 0)
            doctype = doc.get("doctype", "N/A")
            authors_merge = "; ".join(authors)
            affiliations_merge = "; ".join(affiliations)
            citations_titles = fetch_titles(citations, headers)
            citations_str = "; ".join(citations_titles)
            
            articles.append((doctype, title, year, authors_merge, affiliations_merge, citation_count, citations_str))
            
            for author, affiliation_list in zip(authors, affiliations):
                individual_affiliations = [affiliation_list.strip()]

                for affiliation in individual_affiliations:
                    cleaned_affiliation = clean_affiliation(affiliation)
                    carnegie_classification = get_carnegie_classification(cleaned_affiliation, carnegie_df)
                    
                    articles_separated.append((doctype, title, year, author, affiliation, citation_count, citations_str))
                    carnegie_classifications.append((title, year, author, cleaned_affiliation, carnegie_classification))
                    
    return articles, articles_separated, carnegie_classifications

def create_plotly_stacked_histogram(df, x_col, y_col, color_col, title, xlabel, ylabel, output_path):
    df = df[(df[color_col] != 'Unknown') & (df[color_col] != 'abstract')]
    df.sort_values(by=x_col, inplace=True)
    unique_colors = px.colors.qualitative.Safe
    fig = px.histogram(df, x=x_col, y=y_col, color=color_col, title=title, labels={x_col: xlabel, y_col: ylabel}, barmode='stack', color_discrete_sequence=unique_colors)
    fig.update_xaxes(type='category', categoryorder='category ascending')
    
    totals = df.groupby(color_col)[y_col].sum().reset_index()
    totals_dict = totals.set_index(color_col).to_dict()[y_col]
    
    sorted_totals = sorted(totals_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_legend = [item[0] for item in sorted_totals if item[1] > 0]
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=32, family='Arial, sans-serif', color='black', weight='bold')
        ),
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.1,  
            xanchor="center", 
            x=0.5
        ),
        font=dict(size=28),
        legend_traceorder='reversed',
    )
    
    for trace in fig.data:
        trace_name = trace.name.split(" (Total")[0]
        total_count = totals_dict.get(trace_name, 0)
        if total_count == 0:
            trace.showlegend = False
        trace.name = f"{trace_name} (Total: {total_count})"
    
    fig.for_each_trace(
        lambda trace: trace.update(
            showlegend=True if trace.name.split(" (Total")[0] in sorted_legend else False
        )
    )

    fig.write_html(output_path)

def create_plotly_histogram(df, x_col, y_col, title, xlabel, ylabel, output_path):
    df = df[(df[x_col] != 'Unknown') & (df[x_col] != 'abstract')]
    df.sort_values(by=x_col, inplace=True)
    unique_colors = px.colors.qualitative.Safe
    fig = px.histogram(df, x=x_col, y=y_col, title=title, labels={x_col: xlabel, y_col: ylabel}, color_discrete_sequence=unique_colors)
    fig.update_xaxes(type='category', categoryorder='category ascending')
    
    totals = df.groupby(x_col)[y_col].sum().reset_index()
    totals_dict = totals.set_index(x_col).to_dict()[y_col]
    
    sorted_totals = sorted(totals_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_legend = [item[0] for item in sorted_totals if item[1] > 0]
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=32, family='Arial, sans-serif', color='black', weight='bold')
        ),
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.1, 
            xanchor="center", 
            x=0.5
        ),
        font=dict(size=28),
        legend_traceorder='reversed',
    )
    
    for trace in fig.data:
        trace_name = trace.name.split(" (Total")[0]
        total_count = totals_dict.get(trace_name, 0)
        trace.name = f"{trace_name} (Total: {total_count})"
    
    fig.for_each_trace(
        lambda trace: trace.update(
            showlegend=True if trace.name.split(" (Total")[0] in sorted_legend else False
        )
    )

    fig.write_html(output_path)

def create_plotly_line_chart(df, x_col, y_col, color_col, title, xlabel, ylabel, output_path):
    df = df[(df[color_col] != 'Unknown') & (df[color_col] != 'Research Institution')]
    df.sort_values(by=x_col, inplace=True)
    unique_colors = px.colors.qualitative.Safe
    fig = go.Figure()

    total_counts = df.groupby(color_col)[y_col].sum().reset_index()
    totals_dict = total_counts.set_index(color_col).to_dict()[y_col]
    
    sorted_totals = sorted(totals_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_legend = [item[0] for item in sorted_totals]

    for i, classification in enumerate(sorted_legend):
        df_subset = df[df[color_col] == classification]
        total_count = totals_dict[classification]
        color = unique_colors[i % len(unique_colors)]
        fig.add_trace(go.Scatter(
            x=df_subset[x_col], 
            y=df_subset[y_col], 
            mode='lines+markers', 
            name=f'{classification} (Total: {total_count})', 
            line=dict(width=5, color=color),
            marker=dict(color='black')
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=32, family='Arial, sans-serif', color='black', weight='bold')
        ),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.1, 
            xanchor="center", 
            x=0.5
        ),
        xaxis=dict(type='category', categoryorder='category ascending'),
        font=dict(size=28),
        legend_traceorder='reversed',
    )
    
    fig.write_html(output_path)


def create_nasa_centers_chart(df, x_col, y_col, title, xlabel, ylabel, output_path):
    df['Affiliation'] = df['Affiliation'].apply(map_nasa_center)
    create_plotly_line_chart(df, x_col, y_col, 'Affiliation', title, xlabel, ylabel, output_path)

def save_to_excel(writer, df, sheet_name):
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    (max_row, max_col) = df.shape
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': [{'header': col} for col in df.columns]})
    for col_num, value in enumerate(df.columns.values):
        max_len = max(df[value].astype(str).map(len).max(), len(str(value))) + 2
        worksheet.set_column(col_num, col_num, max_len)

def fill_empty_affiliations(df_combined):
    for index, row in df_combined.iterrows():
        if pd.isna(row['Affiliations']) or row['Affiliations'] == "":
            authors = row['Authors'].split("; ")
            for author in authors:
                author_affiliations = df_combined[df_combined['Authors'].str.contains(author) & df_combined['Affiliations'].notna()]['Affiliations'].unique()
                if len(author_affiliations) > 0:
                    df_combined.at[index, 'Affiliations'] = "; ".join(author_affiliations)
                    break
    return df_combined

def merge_excel_files(input_dir, output_dir):
    sheet_data_dict = {}
    for file in os.listdir(input_dir):
        if file.endswith(".xlsx"):
            file_path = os.path.join(input_dir, file)
            data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            for sheet_name, sheet_data in data.items():
                if sheet_name not in sheet_data_dict:
                    sheet_data_dict[sheet_name] = []
                sheet_data_dict[sheet_name].append(sheet_data)
    
    merged_file_path = os.path.join(output_dir, "TotalCubeSatData.xlsx")
    with pd.ExcelWriter(merged_file_path, engine='xlsxwriter') as writer:
        for sheet_name, data_frames in sheet_data_dict.items():
            merged_data = pd.concat(data_frames, ignore_index=True)
            save_to_excel(writer, merged_data, sheet_name)

    charts_output_dir = os.path.join(output_dir, "Chart Data")
    os.makedirs(charts_output_dir, exist_ok=True)

    for sheet_name, data_frames in sheet_data_dict.items():
        merged_df = pd.concat(data_frames, ignore_index=True)
        if sheet_name == 'Combined_Data':
            df_publication_type = merged_df[["Publication Type", "Year Published"]]
            df_year_published = merged_df[["Year Published"]]
            df_citations_by_year = merged_df[["Year Published", "Total Citations"]]

            df_publication_type_long = df_publication_type.pivot_table(index='Year Published', columns='Publication Type', aggfunc='size', fill_value=0).reset_index().melt(id_vars='Year Published')
            create_plotly_stacked_histogram(df_publication_type_long, 'Year Published', 'value', 'Publication Type', 'Publication Type By Year', 'Year', 'Count', os.path.join(charts_output_dir, 'publication_type_by_year.html'))

            df_year_published_long = df_year_published.pivot_table(index='Year Published', aggfunc='size').reset_index().rename(columns={0: 'Count'})
            create_plotly_histogram(df_year_published_long, 'Year Published', 'Count', 'Publications Per Year', 'Year Published', 'Count', os.path.join(charts_output_dir, 'publications_per_year.html'))

            df_citations_by_year_long = df_citations_by_year.groupby('Year Published').sum().reset_index()
            create_plotly_histogram(df_citations_by_year_long, 'Year Published', 'Total Citations', 'Total Citations Per Year', 'Year Published', 'Total Citations', os.path.join(charts_output_dir, 'citations_per_year.html'))

            df_publication_citations = merged_df.groupby(['Year Published', 'Publication Type'])['Total Citations'].sum().reset_index()
            create_plotly_stacked_histogram(df_publication_citations, 'Year Published', 'Total Citations', 'Publication Type', 'Citations Per Publication Type By Year', 'Year', 'Total Citations', os.path.join(charts_output_dir, 'citations_per_pub_type_by_year.html'))

        if sheet_name == 'Carnegie_Class':
            df_classification_long = merged_df.groupby(['Year', 'Carnegie Classification']).size().unstack().fillna(0).reset_index().melt(id_vars='Year')
            create_plotly_line_chart(df_classification_long, 'Year', 'value', 'Carnegie Classification', 'Authors With Carnegie Classifications By Year', 'Year', 'Count', os.path.join(charts_output_dir, 'carnegie_classifications_by_year.html'))

        if sheet_name == 'Authors_Affiliations':
            if 'Affiliation' in merged_df.columns:
                df_nasa_centers = merged_df[merged_df["Affiliation"].str.contains("nasa", case=False, na=False)][["Year Published", "Affiliation"]]
                df_nasa_centers['Affiliation'] = df_nasa_centers['Affiliation'].apply(clean_affiliation)
                df_nasa_centers['Affiliation'] = df_nasa_centers['Affiliation'].apply(map_nasa_center)

                df_nasa_centers_line = df_nasa_centers.groupby(['Year Published', 'Affiliation']).size().unstack().fillna(0).reset_index().melt(id_vars='Year Published', var_name='Affiliation', value_name='Count')
                create_plotly_line_chart(df_nasa_centers_line, 'Year Published', 'Count', 'Affiliation', 'Authors Associated With NASA Centers By Year', 'Year', 'Count', os.path.join(charts_output_dir, 'authors_nasa_centers_by_year.html'))

def scrape_data(entry_keyword, var_include_cubesat):
    title_keyword = entry_keyword.get().strip().lower()
    include_cubesat = var_include_cubesat.get()

    if not title_keyword:
        messagebox.showerror("Input Error", "Please enter a scraping keyword.")
        return
    
    global carnegie_df
    if carnegie_df is None or carnegie_df.empty:
        if 'carnegie_excel_path' in settings:
            try:
                carnegie_df = pd.read_excel(settings['carnegie_excel_path'], usecols=[0, 1], engine='openpyxl')
                carnegie_df.columns = ['Institution', 'Carnegie Classification']
            except Exception as e:
                messagebox.showerror("Error", f"Error loading Carnegie Excel: {e}")
                return
        else:
            excel_path = filedialog.askopenfilename(title="Select Carnegie Excel File", filetypes=[("Excel Files", "*.xlsx")])
            if not excel_path:
                return
            try:
                carnegie_df = pd.read_excel(excel_path, usecols=[0, 1], engine='openpyxl')
                carnegie_df.columns = ['Institution', 'Carnegie Classification']
                settings['carnegie_excel_path'] = excel_path
                save_settings(settings)
            except Exception as e:
                messagebox.showerror("Error", f"Error loading Carnegie Excel: {e}")
                return

    root.update()

    start_time = time.time()
    query = f'abstract:("{title_keyword}" OR "({title_keyword})")'
    if include_cubesat:
        query += ' AND abstract:("cubesat" OR "cubesats")'
    r = requests.get(f'https://api.adsabs.harvard.edu/v1/search/query?q={query}&fl=bibcode&rows=1000', headers={"Authorization": "Bearer " + token})
    time.sleep(1)
    bibcodes = [ii["bibcode"] for ii in r.json()["response"]["docs"]]

    if not bibcodes:
        query = f'title:("{title_keyword}" OR "({title_keyword})")'
        r = requests.get(f'https://api.adsabs.harvard.edu/v1/search/query?q={query}&fl=bibcode&rows=1000', headers={"Authorization": "Bearer " + token})
        time.sleep(1)
        bibcodes = [ii["bibcode"] for ii in r.json()["response"]["docs"]]

    articles, articles_separated, carnegie_classifications = get_article_info(bibcodes, token, carnegie_df, skip_no_abstract=include_cubesat)
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    messagebox.showinfo("Scraping Completed", f"Fetched {len(articles)} articles in {int(minutes)} minutes and {int(seconds)} seconds.")

    title_keyword_clean = re.sub(r'\W+', '_', title_keyword)
    keyword_data_dir = os.path.join(data_dir, f"{title_keyword_clean}_Data")
    os.makedirs(keyword_data_dir, exist_ok=True)
    
    output_path = os.path.join(keyword_data_dir, f"{title_keyword_clean}_Data.xlsx")
    input_copy_path = os.path.join(input_dir, f"{title_keyword_clean}_Data.xlsx")
    charts_output_dir = os.path.join(keyword_data_dir, "Chart Data")
    os.makedirs(charts_output_dir, exist_ok=True)

    df_combined = pd.DataFrame(articles, columns=["Publication Type", "Article Title", "Year Published", "Authors", "Affiliations", "Total Citations", "Article Citations"])
    df_separated = pd.DataFrame(articles_separated, columns=["Publication Type", "Article Title", "Year Published", "Author", "Affiliation", "Total Citations", "Article Citations"])
    df_classification = pd.DataFrame(carnegie_classifications, columns=["Article Title", "Year", "Author", "Affiliation", "Carnegie Classification"])

    df_combined['Affiliations'] = df_combined['Affiliations'].apply(unescape)
    df_separated['Affiliation'] = df_separated['Affiliation'].apply(unescape)

    df_combined = fill_empty_affiliations(df_combined)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        save_to_excel(writer, df_combined, 'Combined_Data')
        save_to_excel(writer, df_separated, 'Authors_Affiliations')
        save_to_excel(writer, df_classification, 'Carnegie_Class')
        save_to_excel(writer, df_combined[["Publication Type", "Year Published"]], 'Pub_Type')
        save_to_excel(writer, df_combined[["Year Published"]], 'Year_Pub')

    with pd.ExcelWriter(input_copy_path, engine='xlsxwriter') as writer:
        save_to_excel(writer, df_combined, 'Combined_Data')
        save_to_excel(writer, df_separated, 'Authors_Affiliations')
        save_to_excel(writer, df_classification, 'Carnegie_Class')
        save_to_excel(writer, df_combined[["Publication Type", "Year Published"]], 'Pub_Type')
        save_to_excel(writer, df_combined[["Year Published"]], 'Year_Pub')

    df_publication_type = df_combined[["Publication Type", "Year Published"]]
    df_year_published = df_combined[["Year Published"]]
    df_citations_by_year = df_combined[["Year Published", "Total Citations"]]

    df_publication_type_long = df_publication_type.pivot_table(index='Year Published', columns='Publication Type', aggfunc='size', fill_value=0).reset_index().melt(id_vars='Year Published')
    create_plotly_stacked_histogram(df_publication_type_long, 'Year Published', 'value', 'Publication Type', 'Publication Type By Year', 'Year', 'Count', os.path.join(charts_output_dir, 'publication_type_by_year.html'))

    df_year_published_long = df_year_published.pivot_table(index='Year Published', aggfunc='size').reset_index().rename(columns={0: 'Count'})
    create_plotly_histogram(df_year_published_long, 'Year Published', 'Count', 'Publications Per Year', 'Year Published', 'Count', os.path.join(charts_output_dir, 'publications_per_year.html'))

    df_citations_by_year_long = df_citations_by_year.groupby('Year Published').sum().reset_index()
    create_plotly_histogram(df_citations_by_year_long, 'Year Published', 'Total Citations', 'Total Citations per Year', 'Year Published', 'Total Citations', os.path.join(charts_output_dir, 'citations_per_year.html'))

    df_classification_long = df_classification.groupby(['Year', 'Carnegie Classification']).size().unstack().fillna(0).reset_index().melt(id_vars='Year')
    create_plotly_line_chart(df_classification_long, 'Year', 'value', 'Carnegie Classification', 'Authors With Carnegie Classifications By Year', 'Year', 'Count', os.path.join(charts_output_dir, 'carnegie_classifications_by_year.html'))

    df_combined['Total Citations'] = df_combined['Total Citations'].astype(int)
    df_publication_citations = df_combined.groupby(['Year Published', 'Publication Type'])['Total Citations'].sum().reset_index()
    create_plotly_stacked_histogram(df_publication_citations, 'Year Published', 'Total Citations', 'Publication Type', 'Citations Per Publication Type By Year', 'Year', 'Total Citations', os.path.join(charts_output_dir, 'citations_per_pub_type_by_year.html'))

    df_nasa_centers = df_separated[df_separated["Affiliation"].str.contains("nasa", case=False, na=False)][["Year Published", "Affiliation"]]
    df_nasa_centers['Affiliation'] = df_nasa_centers['Affiliation'].apply(clean_affiliation)
    df_nasa_centers['Affiliation'] = df_nasa_centers['Affiliation'].apply(map_nasa_center)

    df_nasa_centers_line = df_nasa_centers.groupby(['Year Published', 'Affiliation']).size().unstack().fillna(0).reset_index().melt(id_vars='Year Published', var_name='Affiliation', value_name='Count')
    create_plotly_line_chart(df_nasa_centers_line, 'Year Published', 'Count', 'Affiliation', 'Authors Associated With NASA Centers By Year', 'Year', 'Count', os.path.join(charts_output_dir, 'authors_nasa_centers_by_year.html'))

def merge_files():
    root.update()
    try:
        merge_excel_files(input_dir, output_dir)
        messagebox.showinfo("Merge Completed", "Merge Completed, Graphs Created.")
    except Exception as e:
        messagebox.showerror("Error", f"Error merging files: {e}")

def main():
    global root
    root = tk.Tk()
    root.title("NASA ADS Scraper")
    root.geometry("600x400")

    lbl_keyword = tk.Label(root, text="Enter Scraping Keyword:")
    lbl_keyword.pack(pady=10)
    entry_keyword = tk.Entry(root, width=50)
    entry_keyword.pack(pady=10)

    var_include_cubesat = tk.BooleanVar()
    chk_include_cubesat = tk.Checkbutton(root, text="Include 'CubeSat' in abstract search", variable=var_include_cubesat)
    chk_include_cubesat.pack(pady=10)

    btn_scrape = tk.Button(root, text="Scrape Data", command=lambda: scrape_data(entry_keyword, var_include_cubesat))
    btn_scrape.pack(pady=10)

    btn_merge = tk.Button(root, text="Merge Files", command=merge_files)
    btn_merge.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()