"""
Controlled sentence mutators for creating labelled hallucination test sets.

Applies deterministic, rule-based perturbations to Indian legal summaries
to produce ground-truth hallucinated sentences across five error types:
  1. DATE_NUMBER: Alters dates, amounts (Rs.), counts, or section numbers.
  2. PARTY_OUTCOME: Swaps appellant/respondent or legal outcomes (allowed/dismissed).
  3. NAME_PLACE: Swaps court names, cities, hospitals, or person names.
  4. NEGATION: Inverts sentence polarity (adds/removes negation).
  5. FABRICATED: Injects a plausible yet completely unsupported legal sentence.
"""

import random
import re
from typing import Dict, List, Optional, Tuple

ERROR_TYPES = [
    "DATE_NUMBER",
    "PARTY_OUTCOME",
    "NAME_PLACE",
    "NEGATION",
    "FABRICATED",
]

# Legal outcome & party antonym mappings (bi-directional or targeted)
PARTY_OUTCOME_PAIRS: List[Tuple[str, str]] = [
    (r"\ballowed\b", "dismissed"),
    (r"\bdismissed\b", "allowed"),
    (r"\ballowing\b", "dismissing"),
    (r"\bdismissing\b", "allowing"),
    (r"\bquashed\b", "upheld"),
    (r"\bupheld\b", "quashed"),
    (r"\bset aside\b", "upheld"),
    (r"\bconvicted\b", "acquitted"),
    (r"\bacquitted\b", "convicted"),
    (r"\bconviction\b", "acquittal"),
    (r"\bacquittal\b", "conviction"),
    (r"\bgranted\b", "denied"),
    (r"\brefused\b", "granted"),
    (r"\brefusing\b", "granting"),
    (r"\brejected\b", "accepted"),
    (r"\baccepted\b", "rejected"),
    (r"\bappellant\b", "respondent"),
    (r"\bpetitioner\b", "respondent"),
    (r"\brespondent\b", "petitioner"),
    (r"\bplaintiff\b", "defendant"),
    (r"\bprosecution\b", "defence"),
]

# High Court & institution swaps
COURT_SWAPS: List[Tuple[str, str]] = [
    ("Delhi High Court", "Bombay High Court"),
    ("Bombay High Court", "Delhi High Court"),
    ("Punjab and Haryana High Court", "Madras High Court"),
    ("Jammu and Kashmir and Ladakh High Court", "Allahabad High Court"),
    ("Supreme Court", "High Court"),
    ("High Court", "Supreme Court"),
]

CITY_SWAPS: List[Tuple[str, str]] = [
    (r"\bDelhi\b", "Mumbai"),
    (r"\bMumbai\b", "Delhi"),
    (r"\bJammu\b", "Chandigarh"),
    (r"\bChandigarh\b", "Jammu"),
    (r"\bSaket Courts\b", "Tis Hazari Courts"),
    (r"\bTis Hazari Courts\b", "Saket Courts"),
]

HOSPITAL_SWAPS: List[Tuple[str, str]] = [
    (r"\bPSRI Hospital\b", "Fortis Hospital"),
    (r"\bSafdarjung Hospital\b", "Apollo Hospital"),
    (r"\bEscorts Hospital\b", "Max Super Speciality Hospital"),
]

JUDGE_SWAPS: List[Tuple[str, str]] = [
    ("Justice Rekha Palli", "Justice Sanjeev Khanna"),
    ("Justice Dinesh Kumar Sharma", "Justice Prathiba M. Singh"),
    ("Justice Waseem Sadiq Nargal", "Justice Ali Mohammad Magrey"),
    ("Justice Bharati Dangre", "Justice Gautam Patel"),
    ("Justice Ritu Bahri", "Justice G.S. Sandhawalia"),
    ("ADJ Dinesh Kumar", "ADJ Ramesh Verma"),
]

# Negation polarity transformations
NEGATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bcannot be simply told\b", "can be simply told"),
    (r"\bcannot be permitted\b", "can be permitted"),
    (r"\bcannot be\b", "can be"),
    (r"\bcan be\b", "cannot be"),
    (r"\bdoes not deem\b", "deems"),
    (r"\bdoes not amount\b", "amounts"),
    (r"\bdoes not\b", "does"),
    (r"\bdid not dispute\b", "vehemently disputed"),
    (r"\bdid not\b", "did"),
    (r"\bdo not\b", "do"),
    (r"\bwould not\b", "would"),
    (r"\bfailed to abide\b", "duly abided"),
    (r"\bfailed to\b", "succeeded to"),
    (r"\bhas not\b", "has"),
    (r"\bhave not\b", "have"),
    (r"\bhad not\b", "had"),
    (r"\bhad no other option\b", "had multiple alternative options"),
    (r"\bno reason\b", "ample reason"),
    (r"\bno power connection\b", "a new power connection"),
    (r"\bwas not\b", "was"),
    (r"\bwas justified\b", "was not justified"),
    (r"\bwere justified\b", "were not justified"),
    (r"\bis not proven\b", "is clearly proven"),
    (r"\bhas thankfully survived\b", "unfortunately succumbed to the disease"),
]

# Curated pool of plausible yet entirely fabricated Indian legal sentences
FABRICATED_TEMPLATES: List[str] = [
    "The High Court directed the petitioner to deposit an exemplary security sum of Rs. 15,00,000 with the Prime Minister's National Relief Fund within four weeks.",
    "The Supreme Court invoked its extraordinary powers under Article 142 of the Constitution to order the immediate disciplinary suspension of the investigating officer.",
    "The Division Bench held that failure to produce the electronic certificate under Section 65B of the Evidence Act was fatal to the prosecution's case.",
    "The court ordered the registration of a First Information Report (FIR) under the Prevention of Corruption Act against the municipal authorities.",
    "The bench observed that commercial arbitral awards must be set aside if there is a patent violation of Section 34 of the Arbitration and Conciliation Act.",
    "The High Court held that non-compliance with the statutory 30-day notice period under Section 138 of the Negotiable Instruments Act invalidates the complaint.",
]


def _match_case(original: str, replacement: str) -> str:
    """Preserve title or lower case of replaced token."""
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


def mutate_date_number(sentence: str, seed: Optional[int] = None) -> Optional[Tuple[str, str, str]]:
    """Mutate dates, monetary amounts (Rs.), counts, or section numbers."""
    rng = random.Random(seed)

    # 1. Indian Currency: Rs. X lakhs / crores or Rs. XX,XX,XXX
    # Match: Rs. 16 lakhs -> Rs. 45 lakhs
    currency_lakh_pattern = re.compile(r"(Rs\.?\s*)(\d+)(\s*(?:lakhs?|crores?|lacs?))", re.IGNORECASE)
    match = currency_lakh_pattern.search(sentence)
    if match:
        old_val = int(match.group(2))
        new_val = old_val + rng.choice([15, 25, 30, 50])
        edited = sentence[:match.start()] + f"{match.group(1)}{new_val}{match.group(3)}" + sentence[match.end():]
        return edited, "DATE_NUMBER", f"Changed amount from {match.group(0)} to {match.group(1)}{new_val}{match.group(3)}"

    # Match: Rs. 24,02,380 or Rs.7,08,500
    currency_numeric_pattern = re.compile(r"(Rs\.?\s*)([0-9]{1,3}(?:,[0-9]{2,3})+)", re.IGNORECASE)
    match = currency_numeric_pattern.search(sentence)
    if match:
        old_str = match.group(2)
        # Substitute a plausible different legal figure
        replacement_amounts = ["50,00,000", "1,50,000", "85,20,000", "12,40,000", "3,75,000"]
        rng.shuffle(replacement_amounts)
        new_amount = replacement_amounts[0]
        if new_amount == old_str:
            new_amount = replacement_amounts[1]
        edited = sentence[:match.start()] + f"{match.group(1)}{new_amount}" + sentence[match.end():]
        return edited, "DATE_NUMBER", f"Altered monetary amount from {match.group(0)} to {match.group(1)}{new_amount}"

    # 2. Date: DD.MM.YYYY or DD/MM/YYYY
    date_pattern = re.compile(r"\b(\d{1,2})([./-])(\d{1,2})\2(\d{4})\b")
    match = date_pattern.search(sentence)
    if match:
        day, sep, month, year = match.groups()
        new_year = str(int(year) - rng.choice([1, 2, 3]))
        new_date = f"{day}{sep}{month}{sep}{new_year}"
        edited = sentence[:match.start()] + new_date + sentence[match.end():]
        return edited, "DATE_NUMBER", f"Shifted date from {match.group(0)} to {new_date}"

    # 3. Section or Article numbers: Section 67A -> Section 420; Art. 226 -> Art. 32
    section_pattern = re.compile(r"\b(Section|Sec\.?|Article|Art\.?)\s+([0-9]{1,3}[A-Z]?)\b", re.IGNORECASE)
    match = section_pattern.search(sentence)
    if match:
        prefix, sec_num = match.groups()
        new_secs = ["420", "302", "376", "138", "34", "142", "21"]
        rng.shuffle(new_secs)
        new_sec = new_secs[0] if new_secs[0] != sec_num else new_secs[1]
        edited = sentence[:match.start()] + f"{prefix} {new_sec}" + sentence[match.end():]
        return edited, "DATE_NUMBER", f"Substituted {match.group(0)} with {prefix} {new_sec}"

    # 4. Standalone Year (e.g. 2021 -> 2018)
    year_pattern = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
    match = year_pattern.search(sentence)
    if match:
        old_year = int(match.group(1))
        new_year = old_year - rng.choice([2, 3, 4])
        edited = sentence[:match.start()] + str(new_year) + sentence[match.end():]
        return edited, "DATE_NUMBER", f"Changed year from {old_year} to {new_year}"

    # 5. Counts / durations (e.g. three weeks -> six months)
    duration_pattern = re.compile(r"\b(three weeks|nine months|four weeks|two years)\b", re.IGNORECASE)
    match = duration_pattern.search(sentence)
    if match:
        durations = ["five months", "six years", "eight weeks", "ten days"]
        new_dur = rng.choice(durations)
        edited = sentence[:match.start()] + new_dur + sentence[match.end():]
        return edited, "DATE_NUMBER", f"Altered duration from {match.group(0)} to {new_dur}"

    return None


def mutate_party_outcome(sentence: str, seed: Optional[int] = None) -> Optional[Tuple[str, str, str]]:
    """Swap legal outcomes or party roles with their legal opposites."""
    rng = random.Random(seed)
    pairs = list(PARTY_OUTCOME_PAIRS)
    rng.shuffle(pairs)

    for pattern, replacement in pairs:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            rep = _match_case(matched_text, replacement)
            # Replace only the first occurrence cleanly
            edited = sentence[:match.start()] + rep + sentence[match.end():]
            return edited, "PARTY_OUTCOME", f"Swapped legal term '{matched_text}' with opposite '{rep}'"

    return None


def mutate_name_place(sentence: str, seed: Optional[int] = None) -> Optional[Tuple[str, str, str]]:
    """Swap courts, cities, hospitals, or judicial names."""
    rng = random.Random(seed)

    # 1. High Courts
    court_pairs = list(COURT_SWAPS)
    rng.shuffle(court_pairs)
    for target, replacement in court_pairs:
        if target in sentence:
            edited = sentence.replace(target, replacement, 1)
            return edited, "NAME_PLACE", f"Substituted court '{target}' with '{replacement}'"

    # 2. Judges / Persons
    judge_pairs = list(JUDGE_SWAPS)
    rng.shuffle(judge_pairs)
    for target, replacement in judge_pairs:
        if target in sentence:
            edited = sentence.replace(target, replacement, 1)
            return edited, "NAME_PLACE", f"Substituted name '{target}' with '{replacement}'"

    # 3. Hospitals
    for pattern, replacement in HOSPITAL_SWAPS:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            edited = sentence[:match.start()] + replacement + sentence[match.end():]
            return edited, "NAME_PLACE", f"Substituted medical facility '{matched_text}' with '{replacement}'"

    # 4. Cities / District Courts
    city_pairs = list(CITY_SWAPS)
    rng.shuffle(city_pairs)
    for pattern, replacement in city_pairs:
        match = re.search(pattern, sentence)
        if match:
            matched_text = match.group(0)
            edited = sentence[:match.start()] + replacement + sentence[match.end():]
            return edited, "NAME_PLACE", f"Substituted location '{matched_text}' with '{replacement}'"

    return None


def mutate_negation(sentence: str, seed: Optional[int] = None) -> Optional[Tuple[str, str, str]]:
    """Invert sentence polarity by adding, removing, or flipping negation markers."""
    rng = random.Random(seed)
    patterns = list(NEGATION_PATTERNS)
    rng.shuffle(patterns)

    for pattern, replacement in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            rep = _match_case(matched_text, replacement)
            edited = sentence[:match.start()] + rep + sentence[match.end():]
            return edited, "NEGATION", f"Inverted polarity: replaced '{matched_text}' with '{rep}'"

    # Fallback heuristic: inject 'not' after common auxiliaries if no specific idiom matched
    aux_pattern = re.compile(r"\b(is|was|were|are|has|have|had|would|could|should)\b(?!\s+not\b)", re.IGNORECASE)
    match = aux_pattern.search(sentence)
    if match:
        matched_text = match.group(0)
        rep = f"{matched_text} not"
        edited = sentence[:match.start()] + rep + sentence[match.end():]
        return edited, "NEGATION", f"Injected negation after auxiliary verb '{matched_text}'"

    return None


def mutate_fabricated(sentence: str, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """Provide a plausible yet completely unsupported legal claim from a curated template pool."""
    rng = random.Random(seed)
    fabricated = rng.choice(FABRICATED_TEMPLATES)
    return fabricated, "FABRICATED", "Injected completely unsupported external statutory/directive claim"


def get_applicable_mutations(sentence: str, seed: Optional[int] = None) -> Dict[str, Tuple[str, str]]:
    """Determine all valid mutations that can be applied to a sentence.

    Returns:
        Dict mapping error_type -> (edited_sentence, description).
    """
    applicable = {}
    m_dn = mutate_date_number(sentence, seed=seed)
    if m_dn:
        applicable["DATE_NUMBER"] = (m_dn[0], m_dn[2])

    m_po = mutate_party_outcome(sentence, seed=seed)
    if m_po:
        applicable["PARTY_OUTCOME"] = (m_po[0], m_po[2])

    m_np = mutate_name_place(sentence, seed=seed)
    if m_np:
        applicable["NAME_PLACE"] = (m_np[0], m_np[2])

    m_neg = mutate_negation(sentence, seed=seed)
    if m_neg:
        applicable["NEGATION"] = (m_neg[0], m_neg[2])

    # FABRICATED is always applicable
    m_fab = mutate_fabricated(sentence, seed=seed)
    applicable["FABRICATED"] = (m_fab[0], m_fab[2])

    return applicable


def mutate_sentence(
    sentence: str,
    target_error: Optional[str] = None,
    seed: Optional[int] = None
) -> Optional[Tuple[str, str, str]]:
    """Apply a mutation to a sentence.

    Args:
        sentence: Input summary sentence.
        target_error: Specific error type to attempt, or None to pick from applicable.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (edited_sentence, error_type, description) or None.
    """
    if not sentence or not sentence.strip():
        return None

    if target_error == "DATE_NUMBER":
        return mutate_date_number(sentence, seed=seed)
    elif target_error == "PARTY_OUTCOME":
        return mutate_party_outcome(sentence, seed=seed)
    elif target_error == "NAME_PLACE":
        return mutate_name_place(sentence, seed=seed)
    elif target_error == "NEGATION":
        return mutate_negation(sentence, seed=seed)
    elif target_error == "FABRICATED":
        return mutate_fabricated(sentence, seed=seed)

    # If no target error specified, pick one of the applicable non-fabricated ones, or fabricated
    applicable = get_applicable_mutations(sentence, seed=seed)
    if not applicable:
        return None

    rng = random.Random(seed)
    # Prefer non-fabricated mutations first
    non_fab = [k for k in applicable.keys() if k != "FABRICATED"]
    chosen_type = rng.choice(non_fab) if non_fab else "FABRICATED"
    edited_text, desc = applicable[chosen_type]
    return edited_text, chosen_type, desc
