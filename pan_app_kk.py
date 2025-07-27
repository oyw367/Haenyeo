import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmaz, mesolve, Qobj
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_ancestry_data(filepath):
    df = pd.read_csv(filepath)
    logger.info(f"Loaded ancestry data with {df.shape[0]} samples.")
    return df

def load_selection_data(filepath):
    df = pd.read_csv(filepath)
    logger.info(f"Loaded selection data with {df.shape[0]} SNPs.")
    return df

def load_genotype_bp_data(filepath):
    df = pd.read_csv(filepath)
    logger.info(f"Loaded genotype-BP data for {df.shape[0]} genotypes.")
    return df

def load_clinvar_data(filepath):
    df = pd.read_csv(filepath)
    logger.info(f"Loaded clinvar data with {df.shape[0]} samples.")
    return df

def filter_selection_snp(selection_df, rsid="rs66930627"):
    snp_row = selection_df[selection_df['rsID'] == rsid]
    if snp_row.empty:
        logger.warning(f"SNP {rsid} not found in selection data.")
        return None
    logger.info(f"Selected SNP {rsid} from selection data.")
    return snp_row.iloc[0]

def analyze_jeju_ancestry(ancestry_df):
    # Focus on individuals labeled as Jeju or Korean Mainlander in Jeju
    jeju_samples = ancestry_df[ancestry_df['New_Population_Label'].str.contains('Jeju')]
    logger.info(f"Found {jeju_samples.shape[0]} samples with Jeju ancestry label.")
    
    # Summarize ancestry proportions
    ancestry_summary = jeju_samples[['CHS_Ancestry', 'JEJ_Ancestry', 'KOR_Ancestry']].describe()
    logger.info(f"Ancestry proportions summary:\n{ancestry_summary}")

    return jeju_samples

def annotation_to_gamma(annotation) -> str:
    if 'nonsense' in annotation or 'stop_gained' in annotation:
        return 'Pathogenic'
    elif 'frameshift' in annotation:
        return 'Likely Pathogenic'
    elif 'missense' in annotation:
        return 'Uncertain significance'
    elif 'synonymous' in annotation:
        return 'Benign'
    else:
        return 'Other'


def map_clinvar_to_gamma(clinvar_df: pd.DataFrame):
    clinvar_df['Group'] = clinvar_df['VEP Annotation'].str.lower().apply(annotation_to_gamma)
    gamma_map = {}
    clinvar_df = clinvar_df[clinvar_df['rsIDs'].notna()]
    clinvar_df = clinvar_df[clinvar_df['rsIDs'] != '']
    for _, row in clinvar_df.iterrows():
        rsid = row['rsIDs']
        classification = row['VEP Annotation']
        gamma = annotation_to_gamma(classification)
        gamma_map[rsid] = gamma
        logger.info(f"Variant {rsid} classified as {classification}, assigned gamma {gamma}")
    return gamma_map


def map_genotype_to_ggamma(genotype_bp_df):
    """
    Map mean diastolic BP to a decoherence rate gamma.
    Higher BP → higher gamma (more decoherence).
    """
    min_bp = genotype_bp_df['Mean_Diastolic_BP (mmHg)'].min()
    max_bp = genotype_bp_df['Mean_Diastolic_BP (mmHg)'].max()
    ggamma_base = 0.05
    ggamma_range = 0.1
    
    ggamma_map = {}
    for _, row in genotype_bp_df.iterrows():
        norm_bp = (row['Mean_Diastolic_BP (mmHg)'] - min_bp) / (max_bp - min_bp) if max_bp > min_bp else 0
        ggamma = ggamma_base + ggamma_range * norm_bp
        genotype = row['Genotype']
        ggamma_map[genotype] = ggamma
        logger.info(f"Genotype {genotype} mean BP {row['Mean_Diastolic_BP (mmHg)']} -> gamma {ggamma:.4f}")
    return ggamma_map

def simulate_coherence(gamma_map, tlist):

    psi0 = (basis(2,0) + basis(2,1)).unit()
    Ha = Qobj([[0,0], [0,0]])
    LP = sigmaz()
    gamma_values = {
    'Pathogenic': 0.2,
    'Likely Pathogenic': 0.15,
    'Uncertain significance': 0.1,
    'Benign': 0.05,
    'Other': 0.07
    }

    result = {}
    for sample, label in gamma_map.items():
        g_val = gamma_values.get(label, gamma_values['Other'])
        cops = [np.sqrt(g_val) * LP]
        resultt = mesolve(Ha, psi0, tlist, cops, [sigmax()])
        result[sample] = np.real(resultt.expect[0])
        logger.info(f"Simulated coherence for sample {sample} with gamma {g_val:.4f}")
    return result
    

def simulate_quantum_coherence(ggamma_map, tlist):
    """
    Simulate quantum coherence decay (expectation of sigma_x) per genotype.
    """
    psi0 = (basis(2,0) + basis(2,1)).unit()  # |+> state
    H = Qobj([[0, 0], [0, 0]])  # no system Hamiltonian
    L = sigmaz()                # dephasing operator
    
    results = {}
    for genotype, ggamma in ggamma_map.items():
        c_ops = [np.sqrt(ggamma) * L]
        result = mesolve(H, psi0, tlist, c_ops, [sigmax()])
        results[genotype] = np.real(result.expect[0])
        logger.info(f"Simulated quantum coherence for genotype {genotype} with gamma {ggamma:.4f}")
    return results

def gene_coherence(clinvar_df, tlist, coherence_result, filename=None):
    groups = clinvar_df['Group'].unique()
    colors = {'Pathogenic': 'red', 'Likely Pathogenic': 'orange', 'Uncertain significance': 'blue', 'Benign': 'green', 'Other': 'gray'}

    plt.figure(figsize=(10,6))
    for idx, row in clinvar_df.iterrows():
         group = row['Group']
         y = row['coherence']
         if isinstance(y, (list, np.ndarray)) and len(y) == len(tlist):
                      plt.plot(tlist, y, color=colors.get(group, 'gray'), alpha=0.4)

    plt.xlabel('Time (arb. units)')
    plt.ylabel('Coherence ⟨σ_x⟩')
        
    plt.title('Clinical dataset relevance with coherence')
    plt.grid(True)
    plt.yscale('log')

    fastest_snp = min(coherence_result, key=lambda k: coherence_result[k][3])
    val = coherence_result[fastest_snp][3]
    plt.annotate(f'Fast decay: {fastest_snp}', xy=(tlist[3], val), xytext=(tlist[3], val*1.5),
             arrowprops=dict(arrowstyle='->'))

    # Legend: just group names
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=group) for group, color in colors.items()]
    plt.legend(handles=legend_elements, title="Group")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        logger.info(f"Gene plot saved to {filename}")
    plt.show()

def plot_coherence(tlist, coherence_data, filename=None):
    plt.figure(figsize=(10,6))
    for genotype in sorted(coherence_data.keys()):
        plt.plot(tlist, coherence_data[genotype], label=f'Genotype {genotype}')
    plt.xlabel('Time (arb. units)')
    plt.ylabel('Quantum Coherence ⟨σ_x⟩')
    plt.title('Quantum Coherence Decay by rs66930627 Genotype\nLinked to Diastolic Blood Pressure')
    plt.grid(True)
    plt.legend(title='Genotype')
    plt.tight_layout()
    plt.subplots_adjust(top = 0.9, bottom = 0.15)
    if filename:
        plt.savefig(filename, dpi=300)
        logger.info(f"Plot saved to {filename}")
    plt.show()

def main():
    # Filepaths — replace with your actual CSV file paths
    ancestry_fp = 'ancestor.csv'
    selection_fp = 'snp.csv'
    genotype_bp_fp = 'blood.csv'
    clinvar_fp = 'jej.csv'

    # Load datasets
    ancestry_df = load_ancestry_data(ancestry_fp)
    selection_df = load_selection_data(selection_fp)
    genotype_bp_df = load_genotype_bp_data(genotype_bp_fp)
    clinvar_df = load_clinvar_data(clinvar_fp)

    print(selection_df.columns.tolist())

    
    # Filter and confirm rs66930627 SNP
    snp_info = filter_selection_snp(selection_df, rsid='rs66930627')
    if snp_info is not None:
        logger.info(f"rs66930627 selection p-value: {snp_info['pvalue']}, association_p-value: {snp_info[ 'Associationpvalue']}")
    
    # Analyze Jeju ancestry
    jeju_samples = analyze_jeju_ancestry(ancestry_df)

    gamma_map = map_clinvar_to_gamma(clinvar_df)
    
    # Map genotype BP to quantum decoherence rate gamma
    ggamma_map = map_genotype_to_ggamma(genotype_bp_df)
    
    # Simulation time points
    tlist = np.linspace(0, 10, 400)
    
    # Simulate quantum coherence decay for each genotype
    coherence_result = simulate_coherence(gamma_map, tlist)
    # Add coherence results (arrays) to the dataframe as a new column
    clinvar_df['coherence'] = clinvar_df['rsIDs'].apply(lambda rsid: coherence_result.get(rsid, np.zeros_like(tlist)))

    coherence_results = simulate_quantum_coherence(ggamma_map, tlist)

        # Correct function call
    gene_coherence(clinvar_df, tlist, coherence_result, filename="coherence_clinva.png")

    # Correct value extraction inside gene_coherence:

    plot_coherence(tlist, coherence_results, filename="quantum_coherence_rs66930627.png")

if __name__ == "__main__":
    main()
