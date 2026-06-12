"""Tests for data.pathway_membership.pathway_category.

The local pathway DB stores many inborn-error (disease) and drug-metabolism
pathways under 'metabolic'. The classifier must override that with name-based
heuristics while trusting an explicit stored 'drug'/'disease'.
"""
from data.pathway_membership import pathway_category, filter_by_category


def test_disease_names_override_stored_metabolic():
    for nm in ['Argininemia', 'Prolinemia Type II', 'Cystinuria',
               'Homocarnosinosis', 'Hypermethioninemia', 'Fructosuria',
               'DOPA-Responsive Dystonia',
               'Congenital Bile Acid Synthesis Defect Type III']:
        assert pathway_category(nm, '', 'metabolic') == 'disease', nm


def test_drug_names_override_stored_metabolic():
    for nm in ['Acetaminophen Metabolism Pathway',
               'Fluoxetine Metabolism Pathway', 'Bivalirudin.']:
        assert pathway_category(nm, '', 'metabolic') == 'drug', nm


def test_real_metabolic_pathways_stay_metabolic():
    for nm in ['Alanine Metabolism', 'Citric Acid Cycle', 'Urea Cycle',
               'Arginine and Proline Metabolism', 'Glycolysis',
               'Fatty Acid Biosynthesis', 'Ammonia Recycling',
               'Beta Oxidation of Very Long Chain Fatty Acids']:
        assert pathway_category(nm, '', 'metabolic') == 'metabolic', nm


def test_explicit_stored_disease_drug_trusted():
    assert pathway_category('Some Pathway', '', 'disease') == 'disease'
    assert pathway_category('Some Pathway', '', 'drug') == 'drug'


def test_filter_by_category_excludes_reclassified():
    table = {
        'Alanine Metabolism':  {'description': '', 'category': 'metabolic'},
        'Argininemia':         {'description': '', 'category': 'metabolic'},
        'Fluoxetine Metabolism Pathway': {'description': '', 'category': 'metabolic'},
    }
    met = filter_by_category(table, 'metabolic')
    assert set(met) == {'Alanine Metabolism'}
    assert set(filter_by_category(table, 'disease')) == {'Argininemia'}
    assert set(filter_by_category(table, 'all')) == set(table)
