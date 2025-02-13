from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

from webdriver_manager.chrome import ChromeDriverManager
import requests
import random


def clean_name(name):
    """Clean and normalize name for search"""
    return ' '.join(name.strip().split())


def get_pubchem_synonyms(pubchem_id):
    """Get all synonyms for a PubChem compound"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/synonyms/JSON"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["InformationList"]["Information"][0]["Synonym"]
        return []
    except Exception as e:
        print(f"Error getting PubChem synonyms: {e}")
        return []


def get_uniprot_synonyms(uniprot_id):
    """Get all names and synonyms for a UniProt protein"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            names = []

            # Get protein names
            if "proteinDescription" in data:
                if "recommendedName" in data["proteinDescription"]:
                    names.append(data["proteinDescription"]["recommendedName"]["fullName"]["value"])
                if "alternativeNames" in data["proteinDescription"]:
                    for alt in data["proteinDescription"]["alternativeNames"]:
                        names.append(alt["fullName"]["value"])

            # Get gene names
            if "genes" in data:
                for gene in data["genes"]:
                    if "geneName" in gene:
                        names.append(gene["geneName"]["value"])

            return names
        return []
    except Exception as e:
        print(f"Error getting UniProt synonyms: {e}")
        return []


def setup_driver():
    """Setup and return Chrome driver with options"""
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')
    # Force English language
    options.add_experimental_option('prefs', {
        'intl.accept_languages': 'en-US,en',
        'profile.default_content_setting_values.cookies': 2,  # Block cookies
        'profile.managed_default_content_settings.javascript': 1,  # Allow JavaScript
        'translate.default_language': 'en',
        'translate_language_settings': {'en': True}
    })

    # Add random user agent
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
    options.add_argument(f'user-agent={random.choice(user_agents)}')

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        print(f"Error setting up driver: {e}")
        return None


def get_google_scholar_results(pubchem_id, uniprot_id="P68133",verbose=False):
    """Use Selenium to search Google Scholar and get number of results"""
    # Get synonyms

    compound_names = get_pubchem_synonyms(pubchem_id)
    if verbose:
        print(f"Found {len(compound_names)} compound synonyms")

    protein_names = get_uniprot_synonyms(uniprot_id)
    if verbose:
        print(f"Found {len(protein_names)} protein names")

    # Clean and normalize names
    compound_names = [clean_name(name) for name in compound_names if name.strip()]
    protein_names = [clean_name(name) for name in protein_names if name.strip()]

    # Take first 3 synonyms
    compound_terms = compound_names
    protein_terms = protein_names

    # Format search query
    compound_part = ' OR '.join(f'"{name}"' for name in compound_terms)
    protein_part = ' OR '.join(f'"{name}"' for name in protein_terms)
    query = f'({compound_part}) AND ({protein_part})'

    driver = None
    try:
        driver = setup_driver()
        if not driver:
            return 0

        # Go to Google Scholar
        driver.get('https://scholar.google.com')

        # Search
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        # Type query slowly like a human
        if verbose:
            print(query)
        for char in query:
            search_box.send_keys(char)

        search_box.send_keys(Keys.RETURN)

        # Get results count
        results_stats = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gs_ab_md"))
        )

        results_text = results_stats.text
        if verbose:
            print(f"Results: {results_text}")
        # enslish ?
        number = results_text.split("תוצאות")[0].strip().replace("כ-", "").replace(",", "")

        return int(number) if number else 0

    except Exception as e:
        print(f"Error during search: {e}")
        return 0

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def main():
    print("=== Google Scholar Automated Search ===")

    # Get IDs from user
    pubchem_id = 25198043
    uniprot_id = "P68133"

    print("\nStarting search process...")
    result_count = get_google_scholar_results(pubchem_id, uniprot_id)

    if result_count:
        print(f"\nTotal number of papers found: {result_count:,}")
    else:
        print("\nNo results found or error occurred")


if __name__ == "__main__":
    main()
