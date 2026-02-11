# DRCcode

- [DHS recode manual](https://dhsprogram.com/pubs/pdf/DHSG4/Recode7_DHS_10Sep2018_DHSG4.pdf#page=90)
- [DRC final report 2023–2024](https://dhsprogram.com/pubs/pdf/FR393/FR393.pdf)

<p align="center">
  <img src="Map_of_the_Democratic_Republic_of_the_Congo_with_Provinces.jpg" style="max-width:400px; width:100%;" />
</p>



## Overview DHS Variables

DHS mixes **(a)** identifiers/survey logistics, **(b)** household background, **(c)** women’s module, and **(d)** child/birth-history “roster” blocks that repeat per child/pregnancy.

Here’s a **short clustering** you can use (with the main prefixes you have in your list):

## 1) IDs, sampling, weights, fieldwork meta
**Purpose:** identify record + design variables for weighted/clustered analysis.  
**Vars:** `CASEID`, `V000–V005`, `V001–V004`, `V021–V023`, `V005/ D005`, `V027–V032`, `V047–V048`, interview timing `V006–V019A`, `V801–V806`, selection flags `V042/V044`.

## 2) Geography & residence / mobility
**Purpose:** where respondent lives, rural/urban, migration.  
**Vars:** `V024–V026`, `V101–V105A`, `V103–V104`, `V139–V141`, `V172–V176`, `V174M/V174Y`, `V175`, `V040`.

## 3) Household socioeconomic status & living conditions
**Purpose:** wealth proxies + infrastructure.  
**Vars:** water/sanitation `V113/V115/V116/V160`, assets `V119–V125/V153/V169A–V171B`, housing materials `V127–V129`, cooking fuel `V161`, household composition `V135–V138`, wealth index `V190–V191A`, plus “all woman factors” `AWFACT*`.

## 4) Demographics & education / media exposure
**Purpose:** basic respondent profile + information access.  
**Vars:** DOB/age `V009–V014`, education `V106–V107/V133/V149`, literacy/media `V155–V159`, religion/ethnicity `V130/V131`.

## 5) Fertility history (summary counts) + pregnancy status
**Purpose:** how many births, living/dead children, current pregnancy.  
**Vars:** `V201–V220`, `V211–V213`, `V214–V227`, terminations `V228–V246`, pregnancy wantedness `V225`, menstrual hygiene items `V247*–V249`.

## 6) Contraception & family planning knowledge / sources / reasons
**Purpose:** FP outcomes + program exposure + reasons for nonuse.  
**Vars:** `V301–V327`, `V359–V367A`, FP media `V384A–V384I`, contacts `V393–V395`, counseling `V3A02–V3A06`, reasons `V3A08*`, EC/injectables `V3A11–V3A14`, plus big “source for non-users” block `V3A00*`.

## 7) Maternal care (ANC, delivery, postnatal) + newborn care
**Purpose:** maternal health service use and quality indicators.  
**Vars:** tetanus `M1*`, ANC providers `M2*`, delivery assistance `M3*`, ANC timing/visits `M13–M15`, delivery `M15/M17`, supplements/malaria `M45–M49*`, facility types `M57*`, postnatal checks `M60–M77A`, immediate newborn practices `MNB*`, respectful care block `MH*`.

## 8) Child feeding & nutrition (for youngest/roster child)
**Purpose:** liquids/foods yesterday, stool disposal, advice.  
**Vars:** `V409–V414*`, `V465`, `V469*`, `V486`, and women diet recall `V471*–V472*`.

## 9) Child vaccination & illness care
**Purpose:** immunization dates + recent diarrhea/fever/cough + treatment sources.  
**Vars:** vaccination roster `H1–H69` (and date splits `*D/*M/*Y`), illness & care-seeking `H11–H47`, malaria meds `H37*`, growth monitoring `H70*`.

## 10) Anthropometrics & biomarkers
**Purpose:** measured health outcomes.  
**Vars:** child anthro & anemia `HW*`, women anthro/anemia `V437–V458`, hemoglobin selection/consent `V042/V452*`.

## 11) Marriage/sexual activity, gender norms, empowerment, work
**Purpose:** union history, sex timing, decision-making, attitudes.  
**Vars:** marriage/union `V501–V513`, sex timing/activity `V525–V538`, fertility preferences `V602–V629`, decision-making & norms `V632/V739/V743*`, wife beating norms `V744*`, assets/rights `V745*`, work `V714–V741`.

## 12) HIV/STI knowledge, testing, partners, condoms
**Purpose:** HIV outcomes + risk behavior.  
**Vars:** `V750–V867*`, partners `V766*–V768*`, condom use `V761–V763`, testing `V781–V865`, stigma/attitudes `V775–V859`.

## 13) Domestic violence module
**Purpose:** controlling behaviors, emotional/physical/sexual violence + help-seeking.  
**Vars:** `D100–D130*`, plus interruption/presence controls `V811–V815*`, DV weight `D005`.

## 14) Chronic disease / mental health / special modules
**Purpose:** non-communicable disease, mental health screening, fistula.  
**Vars:** `CHD*`, `MTH*`, `FI*`, plus country-specific `S*`, `SD/SM/SY*`.

---

### “Roster” blocks (why it explodes)
- **Birth history** repeats per child: `B*` (+ `BIDX/BORD/...`)  
- **Pregnancy history** repeats per pregnancy: `P*` (+ `PIDX/PORD/...`)  
- **Child health** repeats per child: `H*`, `HW*`, `ML*`  

So you can treat them as **separate child-level tables** linked back to `CASEID` (and sometimes child line number) rather than “more columns”.

If you tell me what your target outcome is (e.g., **wealth**, **contraceptive use**, **child stunting**, **ANC quality**, **IPV**), I can suggest a *minimal* variable subset per cluster in ~10–30 vars.
