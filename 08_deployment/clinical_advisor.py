# -*- coding: utf-8 -*-
"""
BilinGuard Clinical Decision Support System (CDSS)  —  v3 (agent pipeline)
=========================================================================

A department-, disease-, and jaundice-status-aware clinical advisor layered on
top of the BilinGuard AI predictions.  It is *not* a diagnostic device: it
translates an AI jaundice read-out (screen / grade / DBIL / IBIL / type) plus
optional laboratory values and an admission diagnosis into triage-relevant,
guideline-anchored suggestions for the clinician.

Layers of the knowledge base
----------------------------
1. ``GUIDELINES``       - registry of domestic (Chinese) & international guideline
                          citations, each referenced by a short ``gid``.
2. ``DEPARTMENTS``      - per-department profile (focus, what to do when jaundice
                          is detected, referral logic, key risks).
3. ``DISEASES``         - disease profiles inferred from the admission diagnosis
                          free text (keyword matching), each carrying jaundice
                          type, work-up, treatment principles, red flags.
4. ``JAUNDICE_TYPES``   - hepatocellular / cholestatic / hemolytic profiles.
5. ``SEVERITY_GUIDANCE``- management cadence per clinical severity 0-3.
6. ``CLINICAL_PATHWAYS``- acute red-flag bundles (ALF, ACLF, cholangitis ...).
7. ``AGENT pipeline``  - v3 five-stage agent layer (perceive / route /
                          reason / critique / explain) that adds a structured,
                          auditable ``agent`` payload to every ``advise()``
                          result: decision mode, triggered rules with GRADE-style
                          strength, self-check confidence, and an execution trace.

Departments covered
-------------------
infectious_hepatology (感染/肝病) · gi (消化内科) · hpb_surgery (肝胆外科/普外科·肝胆胰) ·
icu (重症医学科) · hematology (血液科) · oncology (肿瘤科) · emergency (急诊科)

Backward compatibility
----------------------
``ClinicalAdvisor(lang='cn')`` and ``advisor.advise(results, lab_values=None)``
keep working unchanged.  Two optional keyword arguments were added:
``department`` and ``diagnosis``.

DISCLAIMER: output is an AI-assisted screening prompt and does NOT replace
clinical judgement or the responsible physician's decision.
"""
import os

# ═════════════════════════════════════════════════════════════
# 0. GUIDELINE REGISTRY
# ═════════════════════════════════════════════════════════════
# Each entry: short id -> (short title EN, full citation EN, full citation CN,
#                          source/issuing body, year).  Used by DISEASES / PATHWAYS.

GUIDELINES = {
    # ---- Chinese domestic guidelines ----
    'CHB_2022': (
        'Chronic Hepatitis B Prevention & Treatment Guideline (2022)',
        'Chinese Society of Hepatology & Chinese Society of Infectious Diseases. '
        'Guidelines for the Prevention and Treatment of Chronic Hepatitis B (Version 2022).',
        '中华医学会肝病学分会, 中华医学会感染病学分会. 慢性乙型肝炎防治指南（2022年版）. '
        '中华肝脏病杂志, 2022.',
        'CSH / CSID', 2022,
    ),
    'CHC_2022': (
        'Hepatitis C Prevention & Treatment Guideline (2022)',
        'Chinese Society of Hepatology & Chinese Society of Infectious Diseases. '
        'Guidelines for the Prevention and Treatment of Hepatitis C (Version 2022).',
        '中华医学会肝病学分会, 中华医学会感染病学分会. 丙型肝炎防治指南（2022年版）.',
        'CSH / CSID', 2022,
    ),
    'LF_2018': (
        'Guideline for Diagnosis and Treatment of Liver Failure (2018)',
        'Chinese Society of Infectious Diseases & Chinese Society of Hepatology. '
        'Guideline for Diagnosis and Treatment of Liver Failure (2018 edition). '
        'Includes the COSSH-ACLF criteria.',
        '中华医学会感染病学分会肝衰竭与人工肝学组, 中华医学会肝病学分会重型肝病与人工肝学组. '
        '肝衰竭诊治指南（2018年版）. 中华肝脏病杂志, 2018.（含 COSSH-ACLF 诊断标准）',
        'CSH / CSID', 2018,
    ),
    'CIRRHOSIS_2019': (
        'Guidelines on the Management of Cirrhosis (2019)',
        'Chinese Society of Hepatology. Guidelines on the Management of Cirrhosis.',
        '中华医学会肝病学分会. 肝硬化诊治指南（2019年）. 中华肝脏病杂志.',
        'CSH', 2019,
    ),
    'DILI_2023': (
        'Guidelines for Diagnosis and Treatment of Drug-induced Liver Injury (2023)',
        'Chinese Society of Hepatology. Guidelines for the Diagnosis and Treatment '
        'of Drug-induced Liver Injury (2023 edition). Uses RUCAM for causality.',
        '中华医学会肝病学分会. 药物性肝损伤诊治指南（2023年版）.（采用 RUCAM 因果评估）',
        'CSH', 2023,
    ),
    'ALD_2018': (
        'Guideline for Alcoholic Liver Disease (2018 update)',
        'Chinese Society of Hepatology / Fatty Liver & Alcoholic Liver Disease Group. '
        'Guidelines for the Prevention and Treatment of Alcoholic Liver Disease (2018 update).',
        '中华医学会肝病学分会脂肪肝和酒精性肝病学组. 酒精性肝病防治指南（2018年更新版）.',
        'CSH', 2018,
    ),
    'AIH_2021': (
        'Consensus on Autoimmune Hepatitis (2021)',
        'Chinese Society of Hepatology. Consensus on the Diagnosis and Treatment '
        'of Autoimmune Hepatitis (2021).',
        '中华医学会肝病学分会. 自身免疫性肝炎诊断和治疗共识（2021年）.',
        'CSH', 2021,
    ),
    'PBC_2021': (
        'Consensus on Primary Biliary Cholangitis (2021)',
        'Chinese Society of Hepatology. Consensus on the Diagnosis and Treatment '
        'of Primary Biliary Cholangitis (2021).',
        '中华医学会肝病学分会. 原发性胆汁性胆管炎的诊断和治疗共识（2021年）.',
        'CSH', 2021,
    ),
    'PSC_2023': (
        'Consensus on Primary Sclerosing Cholangitis (2023)',
        'Chinese Society of Hepatology. Consensus on the Diagnosis and Treatment '
        'of Primary Sclerosing Cholangitis (2023).',
        '中华医学会肝病学分会. 原发性硬化性胆管炎诊断及治疗共识（2023年）.',
        'CSH', 2023,
    ),
    'ACUTE_BILIARY_2021': (
        'Diagnosis & Treatment of Acute Biliary Infection (2021)',
        'Chinese Society of Surgery, Society of Biliary Surgery. Guidelines for the '
        'Diagnosis and Treatment of Acute Biliary Tract Infections (2021).',
        '中华医学会外科学分会胆道外科学组. 急性胆道系统感染的诊断和治疗指南（2021年）.',
        'CSS - Biliary Surgery', 2021,
    ),
    'HCC_2024': (
        'Guidelines for Diagnosis and Treatment of Primary Liver Cancer (2024)',
        'National Health Commission of China. Guidelines for the Diagnosis and '
        'Treatment of Primary Liver Cancer (2024 edition). Uses CNLC staging.',
        '国家卫生健康委医政司. 原发性肝癌诊疗指南（2024年版）.（采用中国肝癌分期 CNLC）',
        'NHC China', 2024,
    ),
    'VARICEAL_2023': (
        'Consensus on Management of Acute Variceal Bleeding (2023)',
        'Chinese Society of Gastroenterology. Consensus on the Management of Acute '
        'Esophagogastric Variceal Bleeding.',
        '中华医学会消化病学分会. 急性食管胃静脉曲张出血诊治专家共识.',
        'CSGE', 2023,
    ),
    # ---- International guidelines ----
    'EASL_CHB_2017': (
        'EASL CPG on Hepatitis B (2017)',
        'European Association for the Study of the Liver. EASL 2017 Clinical Practice '
        'Guidelines on the management of hepatitis B virus infection. J Hepatol 2017.',
        '欧洲肝脏研究学会 (EASL). EASL 2017 慢性乙型肝炎管理临床实践指南. J Hepatol, 2017.',
        'EASL', 2017,
    ),
    'AASLD_CHB_2024': (
        'AASLD Practice Guidance on Chronic Hepatitis B (2024)',
        'American Association for the Study of Liver Diseases. AASLD 2024 Practice '
        'Guidance on the evaluation and management of chronic hepatitis B. Hepatology 2024.',
        '美国肝病研究学会 (AASLD). 慢性乙型肝炎评估与管理实践指南（2024）.',
        'AASLD', 2024,
    ),
    'EASL_DECOMP_2018': (
        'EASL CPG on Decompensated Cirrhosis (2018, rev. 2024)',
        'EASL. Clinical Practice Guidelines for the management of decompensated '
        'cirrhosis (2018; revised 2024).',
        'EASL. 失代偿期肝硬化管理临床实践指南（2018，2024修订）.',
        'EASL', 2024,
    ),
    'APASL_ACLF_2019': (
        'APASL ACLF Consensus (2019)',
        'Asian Pacific Association for the Study of the Liver. APASL clinical practice '
        'guidelines on acute-on-chronic liver failure (2019 update).',
        '亚太肝病研究学会 (APASL). 慢加急性肝衰竭临床实践指南（2019更新）.',
        'APASL', 2019,
    ),
    'EASL_CLIF_2013': (
        'EASL-CLIF Consortium / CANONIC (2013)',
        'EASL-CLIF Consortium. Definition of acute-on-chronic liver failure (CANONIC '
        'study) and CLIF-C ACLF / CLIF-C OF organ-failure scoring. Gut & J Hepatol, 2013-2014.',
        'EASL-CLIF 联盟. CANONIC 研究定义慢加急性肝衰竭；CLIF-C ACLF / CLIF-C OF 器官衰竭评分（2013-2014）.',
        'EASL-CLIF', 2013,
    ),
    'AASLD_PSC_2024': (
        'AASLD Practice Guidance on PSC (2024)',
        'AASLD. Practice Guidance on Primary Sclerosing Cholangitis (2024).',
        'AASLD. 原发性硬化性胆管炎实践指南（2024）.',
        'AASLD', 2024,
    ),
    'ACG_ALD_2018': (
        'ACG Alcoholic Liver Disease Guideline (2018)',
        'American College of Gastroenterology. Alcoholic Liver Disease Guideline (2018).',
        '美国胃肠病学院 (ACG). 酒精性肝病指南（2018）.',
        'ACG', 2018,
    ),
    'TOKYO_TG18': (
        'Tokyo Guidelines (TG18) for Acute Cholangitis/Cholecystitis',
        'Yokoe M et al. Tokyo Guidelines 2018 (TG18): diagnostic criteria and severity '
        'grading of acute cholangitis/cholecystitis. J Hepatobiliary Pancreat Sci 2018.',
        '东京指南 2018 (TG18)：急性胆管炎/胆囊炎诊断标准与严重程度分级. '
        'J Hepatobiliary Pancreat Sci, 2018.',
        'Tokyo Guidelines', 2018,
    ),
    'KINGS_COLLEGE_ALF': (
        "King's College Criteria for Acute Liver Failure",
        "O'Grady JG et al. King's College Criteria for liver transplantation in "
        "paracetamol (APAP) and non-APAP acute liver failure. Defines ALF transplant "
        "futility thresholds.",
        "King 学院标准：对乙酰氨基酚(APAP)及非APAP急性肝衰竭肝移植指征判定.",
        "King's College Hospital", 1989,  # validated, still in use
    ),
    'AASLD_ALF_2023': (
        'AASLD Guidance on Acute Liver Failure (2023)',
        'AASLD. Acute Liver Failure: Pathogenesis, Clinical Manifestations, Management '
        '(2023 Practice Guidance). Recommends NAC for non-APAP ALF.',
        'AASLD. 急性肝衰竭管理实践指南（2023）：推荐非对乙酰氨基酚ALF使用NAC.',
        'AASLD', 2023,
    ),
    'MELD_3_0': (
        'MELD-Na / MELD 3.0 for Transplant Prioritization',
        'OPTN/UNOS. MELD-Na and MELD 3.0 (2023) for deceased-donor liver transplant '
        'allocation priority.',
        'OPTN/UNOS. MELD-Na 及 MELD 3.0（2023）用于肝移植分配优先级.',
        'OPTN / UNOS', 2023,
    ),
    'CHILD_PUGH': (
        'Child-Pugh Classification',
        'Pugh RNH et al. Child-Pugh (CTP) score: bilirubin, albumin, INR, ascites, '
        'encephalopathy → A/B/C severity of chronic liver disease.',
        'Child-Pugh 分级 (CTP)：胆红素/白蛋白/INR/腹水/肝性脑病 → A/B/C.',
        'Classical', 1973,
    ),
    # ── Child-Pugh & MELD evidence (international) ──
    'MELD_ORIGINAL_2001': (
        'MELD Score (Mayo, Kamath 2001)',
        'Kamath PS, Wiesner RH, Malinchoc M, et al. A model to predict survival in patients '
        'with end-stage liver disease. Hepatology 2001;33(2):464-470. The modified MELD used '
        'clinically (bilirubin, INR, creatinine).',
        'Kamath PS 等. 终末期肝病生存预测模型（改良MELD）. Hepatology 2001;33(2):464-470. '
        '临床常用MELD（胆红素、INR、肌酐）.',
        'Mayo / AASLD', 2001,
    ),
    'MELD_NA_2006': (
        'MELD-Na (Biggins 2006)',
        'Biggins SW, Kim WR, Terrault NA, et al. Evidence-based incorporation of serum sodium '
        'concentration into MELD (MELD-Na). Gastroenterology 2006. Sodium improves prediction '
        'of wait-list mortality.',
        'Biggins SW 等. 基于证据将血清钠纳入MELD（MELD-Na）. Gastroenterology 2006. '
        '血钠提升等待期死亡预测准确性.',
        'AASLD / OPTN', 2006,
    ),
    'MELD_3_0_2021': (
        'MELD 3.0 (Kim 2021)',
        'Kim WR, Mannalithara A, Cholankeril G, et al. MELD 3.0: The Model for End-stage Liver '
        'Disease Updated for the Current Era (adds albumin and sex). J Hepatol / adopted by '
        'OPTN from 2023.',
        'Kim WR 等. MELD 3.0：适应当代的终末期肝病模型（新增白蛋白与性别）. J Hepatol, '
        '2023年起被OPTN采用.',
        'OPTN / UNOS', 2021,
    ),
    'ALBI_2015': (
        'ALBI Grade (Johnson 2015)',
        'Johnson PJ, Berhane S, Kagebayashi C, et al. Assessment of liver function in patients '
        'with hepatocellular carcinoma: a new evidence-based approach — the ALBI grade '
        '(albumin + bilirubin). J Clin Oncol 2015;33(6):550-558. Objective alternative to '
        'Child-Pugh for HCC liver reserve.',
        'Johnson PJ 等. 肝癌患者肝功能评估的新方法——ALBI 分级（白蛋白+胆红素）. '
        'J Clin Oncol 2015;33(6):550-558. HCC 肝储备的客观替代，优于 Child-Pugh 主观性.',
        'ILCA / JCO', 2015,
    ),
    'OKUDA_1985': (
        'Okuda Staging (Historical HCP)',
        'Okuda K et al. Natural history of hepatocellular carcinoma and prognosis in relation '
        'to treatment (Okuda stage). Cancer 1985. Pre-Child-Pugh/MELD HCC prognostic system.',
        'Okuda K 等. 肝细胞癌自然史与预后（Okuda 分期）. Cancer 1985. 早于Child-Pugh/MELD的HCC预后系统.',
        'Classical', 1985,
    ),
    'CHILD_TURCOTTE_1964': (
        'Child-Turcotte Original (1964)',
        'Child CG, Turcotte JG. Surgery and portal hypertension (original Child classification). '
        'Major Problems in Clinical Surgery 1964. Historical basis of the modern CTP score.',
        'Child CG, Turcotte JG. 外科与门脉高压（原始Child分级）. 1964. 现代CTP评分的历史基础.',
        'Classical', 1964,
    ),
    # ── Child-Pugh & MELD evidence (Chinese) ──
    'CHINA_LIVER_TRANSPLANT_2018': (
        'Chinese Consensus on Liver-Transplant Recipient Selection (2018)',
        'Chinese Society of Organ Transplantation / Chinese Medical Association Organ '
        'Transplantation Physician Branch. Expert consensus on selection of liver transplant '
        'recipients (uses MELD/MELD-Na for donor allocation in China).',
        '中国医师协会器官移植医师分会, 中华医学会器官移植学分会. 肝移植受者选择专家共识 '
        '（采用MELD/MELD-Na进行供肝分配）.',
        'CMA / CSOT', 2018,
    ),
    'CHINA_CLTR_MELD': (
        'China Liver Transplant Registry (CLTR) — MELD-based allocation',
        'China Liver Transplant Registry (CLTR). National liver-transplant allocation and '
        'outcome data using MELD-based scoring for end-stage liver disease in China.',
        '中国肝移植注册系统(CLTR). 以MELD评分为基础的终末期肝病肝移植分配与预后数据.',
        'CLTR', 2024,
    ),
    'CHINA_PORTAL_HYPERTENSION_2023': (
        'Chinese Guideline on Portal Hypertension & Variceal Bleeding (2023)',
        'Chinese Society of Gastroenterology / Hepatology. Guideline on diagnosis and treatment '
        'of portal hypertension and variceal bleeding (stratifies by Child-Pugh).',
        '中华医学会消化病学分会/肝病学分会. 门静脉高压及食管胃静脉曲张出血诊治指南 '
        '（按Child-Pugh分层）.',
        'CSGE / CSH', 2023,
    ),
    'WORLD_GALLSTONE_2022': (
        'World Gastroenterology Organisation Gallstone Guideline (2022)',
        'WGO. Gallstones: Epidemiology, Diagnosis and Management (2022).',
        '世界胃肠病组织 (WGO). 胆结石流行病学、诊断与管理（2022）.',
        'WGO', 2022,
    ),
    'EASL_PBC_2017': (
        'EASL CPG on PBC (2017)',
        'EASL. Clinical Practice Guidelines on PBC (2017).',
        'EASL. 原发性胆汁性胆管炎临床实践指南（2017）.',
        'EASL', 2017,
    ),
    'HE_ENCEPHALOPATHY_2014': (
        'EASL/AASLD Hepatic Encephalopathy Guidance',
        'Vilstrup H et al. HE in chronic liver disease: 2014 Practice Guideline '
        '(AASLD/EASL). West Haven grading of encephalopathy.',
        'AASL/EASL 肝性脑病实践指南（2014）：West Haven 分级.',
        'AASLD / EASL', 2014,
    ),
    # ---- Expanded hepatology ----
    'EASL_DILI_2019': (
        'EASL CPG on Drug-induced Liver Injury (2019)',
        'EASL. Clinical Practice Guidelines on drug-induced liver injury (2019).',
        'EASL. 药物性肝损伤临床实践指南（2019）.',
        'EASL', 2019,
    ),
    'EASL_ALD_2018': (
        'EASL CPG on Alcohol-related Liver Disease (2018)',
        'EASL. Clinical Practice Guidelines on alcohol-related liver disease (2018).',
        'EASL. 酒精相关肝病临床实践指南（2018）.',
        'EASL', 2018,
    ),
    'EASL_ARD_2024': (
        'EASL CPG on Alcohol-related Liver Disease (2024 update)',
        'EASL. CPG on alcohol-related liver disease (2024 update) — alcohol-associated '
        'hepatitis management.',
        'EASL. 酒精相关肝病临床实践指南（2024更新）——酒精性肝炎处理.',
        'EASL', 2024,
    ),
    'EASL_AIH_2015': (
        'EASL CPG on Autoimmune Hepatitis (2015)',
        'EASL. Clinical Practice Guidelines on autoimmune hepatitis (2015).',
        'EASL. 自身免疫性肝炎临床实践指南（2015）.',
        'EASL', 2015,
    ),
    'AASLD_AIH_2019': (
        'AASLD Practice Guidance on Autoimmune Hepatitis (2019)',
        'AASLD. Practice Guidance and Guidelines for AIH Diagnosis and Management (2019).',
        'AASLD. 自身免疫性肝炎诊断与管理实践指南（2019）.',
        'AASLD', 2019,
    ),
    'EASL_PSC_2022': (
        'EASL CPG on Sclerosing Cholangitis (2022)',
        'EASL. Clinical Practice Guidelines on sclerosing cholangitis (2022).',
        'EASL. 硬化性胆管炎临床实践指南（2022）.',
        'EASL', 2022,
    ),
    'EASL_HEV_2018': (
        'EASL Recommendations on HEV Infection (2018)',
        'EASL. Recommendations on treatment of hepatitis E virus infection (2018).',
        'EASL. 戊型肝炎病毒感染治疗建议（2018）.',
        'EASL', 2018,
    ),
    'HCV_GUIDANCE_2023': (
        'AASLD-IDSA HCV Guidance (living)',
        'AASLD-IDSA. Recommendations for testing, management and treatment of hepatitis C '
        '(living guidance, updated 2023). DAA-based curative regimens.',
        'AASLD-IDSA. 丙型肝炎检测、管理与治疗推荐（动态更新2023）——DAA治愈方案.',
        'AASLD-IDSA', 2023,
    ),
    'EASL_NIT_2021': (
        'EASL CPG on Non-invasive Tests for Liver Disease (2021)',
        'EASL. CPG on non-invasive tests for evaluation of liver disease severity and '
        'prognosis (2021).',
        'EASL. 评估肝病严重程度与预后的无创检测临床实践指南（2021）.',
        'EASL', 2021,
    ),
    'EASL_HCC_2024': (
        'EASL CPG on Management of HCC (2024, BCLC)',
        'EASL. Clinical Practice Guideline on management of hepatocellular carcinoma '
        '(2024). Incorporates Barcelona Clinic Liver Cancer (BCLC) staging.',
        'EASL. 肝细胞癌管理临床实践指南（2024）——含 BCLC 分期.',
        'EASL', 2024,
    ),
    'BCLC_2022': (
        'BCLC 2022 Update',
        'Reig M et al. BCLC strategy for HCC prognosis prediction and treatment '
        'recommendation: 2022 update. J Hepatol.',
        'BCLC 2022 更新：肝细胞癌预后预测与治疗推荐（Reig等, J Hepatol 2022）.',
        'BCLC group', 2022,
    ),
    'AASLD_HCC_2023': (
        'AASLD Practice Guidance on HCC (2023)',
        'AASLD. Practice Guidance on surveillance, diagnosis and treatment of HCC (2023).',
        'AASLD. 肝细胞癌监测、诊断与治疗实践指南（2023）.',
        'AASLD', 2023,
    ),
    'EASL_WILSON_2012': (
        'EASL CPG on Wilson Disease (2012)',
        'EASL. Clinical Practice Guidelines on Wilson\'s disease (2012).',
        'EASL. 肝豆状核变性临床实践指南（2012）.',
        'EASL', 2012,
    ),
    'EASL_VASCULAR_2016': (
        'EASL CPG on Vascular Liver Disorders (2016)',
        'EASL. Clinical Practice Guidelines on vascular liver disorders: Budd-Chiari, '
        'portal vein thrombosis, etc. (2016).',
        'EASL. 肝血管疾病临床实践指南（2016）：布加综合征、门脉血栓等.',
        'EASL', 2016,
    ),
    'EASL_COAG_2022': (
        'EASL CPG on Coagulation in Liver Disease (2022)',
        'EASL. Clinical Practice Guidelines on coagulation disorders in liver disease (2022).',
        'EASL. 肝病凝血障碍临床实践指南（2022）.',
        'EASL', 2022,
    ),
    'SURVIVING_SEPSIS_2021': (
        'Surviving Sepsis Campaign Guidelines (2021)',
        'Evans L et al. Surviving Sepsis Campaign International Guidelines for management '
        'of sepsis and septic shock (2021). SCCM/ESICM.',
        '拯救脓毒症运动国际指南（2021）：脓毒症与感染性休克管理. SCCM/ESICM.',
        'SCCM / ESICM', 2021,
    ),
    'ACR_JAUNDICE': (
        'ACR Appropriateness Criteria: Jaundice',
        'American College of Radiology. ACR Appropriateness Criteria: Jaundice — best '
        'imaging evaluation of new-onset jaundice (ultrasound first-line, then MRCP/CT).',
        '美国放射学会(ACR)适宜性标准：黄疸——首选超声，继而MRCP/CT.',
        'ACR', 2024,
    ),
    # ---- Obstetrics / pregnancy-related liver disease ----
    'RCOG_ICP_2023': (
        'RCOG Green-top No.43: Obstetric Cholestasis (ICP) (2023)',
        'RCOG. Green-top Guideline No. 43: Intrahepatic Cholestasis of Pregnancy '
        '(Obstetric Cholestasis) (2023 update).',
        '英国皇家妇产科医学院(RCOG) 绿顶指南43号：妊娠期肝内胆汁淤积症（2023更新）.',
        'RCOG', 2023,
    ),
    'ACOG_ICP_2023': (
        'ACOG Clinical Consensus No.6: ICP (2023)',
        'ACOG. Clinical Consensus No. 6: Intrahepatic Cholestasis of Pregnancy (2023).',
        '美国妇产科医师学会(ACOG) 临床共识第6号：妊娠期肝内胆汁淤积症（2023）.',
        'ACOG', 2023,
    ),
    'ICP_CHINA_2024': (
        'Chinese Consensus on ICP (2024)',
        'Chinese Society of Obstetrics and Gynecology, Obstetrics Group. Clinical '
        'Guideline for Diagnosis and Management of Intrahepatic Cholestasis of Pregnancy (2024).',
        '中华医学会妇产科学分会产科学组. 妊娠期肝内胆汁淤积症临床诊治和管理指南（2024）.',
        'CSOB-GYN', 2024,
    ),
    'ACOG_PREECLAMPSIA_2020': (
        'ACOG Practice Bulletin 222: Gestational Hypertension & Preeclampsia (incl. HELLP) (2020)',
        'ACOG. Practice Bulletin No. 222: Gestational Hypertension and Preeclampsia (2020). '
        'Covers HELLP syndrome.',
        'ACOG 实践公告222号：妊娠期高血压与子痫前期（2020），含 HELLP 综合征.',
        'ACOG', 2020,
    ),
    'CHINA_HDP_2020': (
        'Chinese Guideline on Hypertensive Disorders of Pregnancy (2020)',
        'Chinese Society of Obstetrics and Gynecology. Guideline for Diagnosis and '
        'Treatment of Hypertensive Disorders of Pregnancy (2020).',
        '中华医学会妇产科学分会. 妊娠期高血压疾病诊治指南（2020）.',
        'CSOB-GYN', 2020,
    ),
    # ---- Nephrology / hepatorenal & TMA ----
    'KDIGO_AKI_2012': (
        'KDIGO AKI Clinical Practice Guideline (2012)',
        'KDIGO. Clinical Practice Guideline for Acute Kidney Injury (2012). Defines '
        'hepatorenal syndrome-AKI (HRS-AKI) criteria.',
        'KDIGO. 急性肾损伤临床实践指南（2012）：定义肝肾综合征-AKI(HRS-AKI)标准.',
        'KDIGO', 2012,
    ),
    'ICA_HRS_2024': (
        'International Club of Ascites: AKI in Cirrhosis Consensus (2024)',
        'International Club of Ascites (ICA). Consensus on acute kidney injury in cirrhosis '
        'and hepatorenal syndrome (2024): albumin + terlipressin, transplant evaluation.',
        '国际腹水俱乐部(ICA). 肝硬化急性肾损伤与肝肾综合征共识（2024）：白蛋白+特利加压素、移植评估.',
        'ICA', 2024,
    ),
    'TMA_HUS_GUIDANCE': (
        'Thrombotic Microangiopathy / HUS Guidance',
        'Clinical guidance on thrombotic microangiopathy (TMA) and haemolytic-uraemic '
        'syndrome (HUS): plasma exchange, complement evaluation, dialysis.',
        '血栓性微血管病(TMA)/溶血尿毒综合征(HUS)临床指导：血浆置换、补体评估、透析.',
        'Haematology / Nephrology', 2020,
    ),
    # ---- Neonatal (reference only — out of scope for adult tool) ----
    'NICE_NEONATAL_JAUNDICE_CG98': (
        'NICE CG98: Jaundice in Newborn Babies (reference)',
        'NICE. Clinical Guideline CG98: Jaundice in newborn babies under 28 days (2010). '
        'REFERENCE ONLY — BilinGuard is an ADULT tool and must NOT be applied to neonates.',
        'NICE 临床指南CG98：新生儿黄疸（≤28天）. 仅供参考——BilinGuard为成人工具，禁用于新生儿.',
        'NICE', 2010,
    ),
    'AAP_HYPERBILI_2022': (
        'AAP Guideline: Neonatal Hyperbilirubinemia (reference)',
        'American Academy of Pediatrics. Clinical Practice Guideline on Management of '
        'Hyperbilirubinemia in the Newborn Infant (2022). REFERENCE ONLY — adult tool.',
        '美国儿科学会(AAP). 新生儿高胆红素血症管理临床实践指南（2022）. 仅供参考——成人工具.',
        'AAP', 2022,
    ),
    # ---- Nutrition (reference) ----
    'ESPEN_LIVER_2019': (
        'ESPEN Guideline: Clinical Nutrition in Liver Disease (2019)',
        'ESPEN. Guideline on clinical nutrition in liver disease (2019).',
        '欧洲临床营养和代谢学会(ESPEN). 肝病临床营养指南（2019）.',
        'ESPEN', 2019,
    ),
    'NRS2002_2003': (
        'NRS-2002 Nutrition Risk Screening (Kondrup 2003)',
        'Kondrup J, Allison SP, Elia M, Vellas B, Plauth M. ESPEN guidelines for '
        'nutrition screening 2002. Clin Nutr 2003;22(4):415-421.',
        'Kondrup J 等. ESPEN 营养风险筛查2002指南（NRS-2002）. Clin Nutr 2003;22(4):415-421.',
        'ESPEN', 2003,
    ),
}


# ═════════════════════════════════════════════════════════════
# 1. DEPARTMENTS  —  who is using BilinGuard, and what they need
# ═════════════════════════════════════════════════════════════

DEPARTMENTS = {
    'infectious_hepatology': {
        'name_en': 'Infectious Disease / Hepatology',
        'name_cn': '感染科 / 肝病科',
        'focus_en': 'Primary work-up and medical management of hepatocellular jaundice, '
                    'viral hepatitis, liver failure and decompensated cirrhosis.',
        'focus_cn': '肝细胞性黄疸、病毒性肝炎、肝衰竭及失代偿期肝硬化的首要病因筛查与内科治疗。',
        'on_jaundice_en': [
            'Confirm the aetiology (viral serology, autoimmune, drug/alcohol history).',
            'Stratify severity (Child-Pugh / MELD-Na / COSSH-ACLF) and screen for organ failure.',
            'Start aetiology-specific therapy early (e.g. NAs for HBV-related ACLF).',
            'Watch for decompensation: encephalopathy, ascites, variceal bleeding, infection (SBP).',
        ],
        'on_jaundice_cn': [
            '明确病因（病毒血清学、自身免疫、药物/酒精史）。',
            '评估严重程度（Child-Pugh / MELD-Na / COSSH-ACLF）并筛查器官衰竭。',
            '尽早启动病因治疗（如HBV相关ACLF立即启动核苷类似物）。',
            '警惕失代偿：肝性脑病、腹水、静脉曲张出血、感染（自发性细菌性腹膜炎）。',
        ],
        'referral_en': 'Refer to HPB surgery if imaging shows biliary obstruction or mass; '
                       'to ICU if organ failure; to transplant centre if ALF/ACLF criteria met.',
        'referral_cn': '影像提示胆道梗阻或占位 → 肝胆外科；出现器官衰竭 → ICU；'
                       '符合ALF/ACLF标准 → 联系肝移植中心。',
        'guidelines': ['CHB_2022', 'CHC_2022', 'LF_2018', 'CIRRHOSIS_2019',
                       'DILI_2023', 'AIH_2021', 'PBC_2021', 'APASL_ACLF_2019',
                       'EASL_DILI_2019', 'EASL_ARD_2024', 'EASL_AIH_2015', 'EASL_NIT_2021'],
    },
    'gi': {
        'name_en': 'Gastroenterology',
        'name_cn': '消化内科',
        'focus_en': 'Diagnostic work-up of mixed-pattern jaundice, ERCP/EUS for '
                    'biliary disease, and endoscopic management of cirrhosis complications.',
        'focus_cn': '混合型黄疸的诊断、胆道疾病的 ERCP/EUS，以及肝硬化并发症的内镜处理。',
        'on_jaundice_en': [
            'First-line abdominal ultrasound ± MRCP to separate obstruction from parenchymal disease.',
            'Distinguish cholestatic vs hepatocellular pattern by ALT/AST vs ALP/GGT.',
            'ERCP for therapeutic biliary decompression (stones, stenting).',
            'Screen oesophageal varices and treat if decompensated.',
        ],
        'on_jaundice_cn': [
            '首选腹部超声±MRCP，鉴别梗阻性与肝实质性疾病。',
            '用 ALT/AST vs ALP/GGT 区分胆汁郁积型与肝细胞型。',
            'ERCP 行胆道减压（取石、支架）。',
            '失代偿者筛查食管胃静脉曲张并处理。',
        ],
        'referral_en': 'To HPB surgery for surgical obstruction / malignancy; to '
                       'infectious disease-hepatology for parenchymal liver disease.',
        'referral_cn': '外科性梗阻/恶性病变 → 肝胆外科；肝实质病变 → 感染/肝病科。',
        'guidelines': ['CIRRHOSIS_2019', 'VARICEAL_2023', 'ACUTE_BILIARY_2021', 'TOKYO_TG18',
                       'ACR_JAUNDICE', 'EASL_NIT_2021', 'EASL_PSC_2022'],
    },
    'hpb_surgery': {
        'name_en': 'Hepatobiliary & Pancreatic Surgery (General Surgery)',
        'name_cn': '肝胆外科 / 普外科（肝胆胰）',
        'focus_en': 'Surgical causes of cholestatic/obstructive jaundice: choledocholithiasis, '
                    'cholangitis, tumours of the biliary tree and pancreatic head.',
        'focus_cn': '胆汁郁积/梗阻性黄疸的外科病因：胆总管结石、胆管炎、胆道及胰头肿瘤。',
        'on_jaundice_en': [
            'Separate surgical obstruction (dilated ducts, direct-bilirubin dominant) from '
            'medical jaundice BEFORE any invasive step.',
            'Correct coagulopathy pre-operatively (INR<1.5; vitamin K, FFP, PCC).',
            'Grade acute cholangitis by Tokyo Guidelines (TG18) → timing of biliary drainage.',
            'For malignant obstruction: stage (CT/MRI) and plan neoadjuvant vs upfront resection.',
        ],
        'on_jaundice_cn': [
            '任何有创操作前，先鉴别外科梗阻（胆管扩张、直接胆红素为主）与内科黄疸。',
            '术前纠正凝血障碍（INR<1.5；维生素K、FFP、PCC）。',
            '按东京指南(TG18)对急性胆管炎分级 → 决定胆道引流时机。',
            '恶性梗阻：分期（CT/MRI），决定新辅助治疗或直接手术切除。',
        ],
        'referral_en': 'To GI/ERCP for endoscopic drainage; to oncology / MDT for '
                       'malignancy; to ICU for post-operative liver failure or sepsis.',
        'referral_cn': '内镜引流 → 消化/ERCP；恶性病变 → 肿瘤科/MDT；术后肝衰或脓毒症 → ICU。',
        'guidelines': ['ACUTE_BILIARY_2021', 'TOKYO_TG18', 'HCC_2024', 'WORLD_GALLSTONE_2022',
                       'ACR_JAUNDICE', 'EASL_COAG_2022'],
    },
    'icu': {
        'name_en': 'Intensive Care (Critical Care)',
        'name_cn': '重症医学科 (ICU)',
        'focus_en': 'Organ support in acute liver failure, ACLF with organ failure, '
                    'severe septic cholangitis and post-major-hepatectomy liver failure.',
        'focus_cn': '急性肝衰竭、伴器官衰竭的ACLF、重症化脓性胆管炎及大肝切除术后肝衰竭的器官支持。',
        'on_jaundice_en': [
            'Compute organ-failure scores hourly: CLIF-C OF / SOFA; trend lactate, INR, ammonia, glucose.',
            'Apply ALF bundle: N-acetylcysteine regardless of aetiology, cerebral oedema vigilance, '
            'haemodynamic and renal support.',
            "Evaluate King's College Criteria and list for transplant if met.",
            'Treat precipitants of ACLF (infection, bleeding, alcohol, hepatotoxic drugs).',
        ],
        'on_jaundice_cn': [
            '每小时计算器官衰竭评分：CLIF-C OF / SOFA；动态监测乳酸、INR、血氨、血糖。',
            '执行ALF集束化方案：不论病因给予NAC，警惕脑水肿，循环与肾脏支持。',
            '评估King学院标准，符合则列入肝移植。',
            '去除ACLF诱因（感染、出血、酒精、肝毒性药物）。',
        ],
        'referral_en': 'To transplant surgery/hepatology for ALF/ACLF; to HPB surgery '
                       'for source control of cholangitis; to infectious disease for sepsis.',
        'referral_cn': 'ALF/ACLF → 肝移植外科/肝病科；胆管炎源头控制 → 肝胆外科；脓毒症 → 感染科。',
        'guidelines': ['LF_2018', 'APASL_ACLF_2019', 'EASL_CLIF_2013',
                       'KINGS_COLLEGE_ALF', 'AASLD_ALF_2023', 'HE_ENCEPHALOPATHY_2014',
                       'SURVIVING_SEPSIS_2021', 'EASL_COAG_2022'],
    },
    'hematology': {
        'name_en': 'Hematology',
        'name_cn': '血液科',
        'focus_en': 'Haemolytic jaundice (indirect-bilirubin dominant): AIHA, G6PD '
                    'deficiency, haemoglobinopathies, microangiopathic and mechanical haemolysis.',
        'focus_cn': '溶血性黄疸（间接胆红素为主）：AIHA、G6PD缺乏、血红蛋白病、'
                    '微血管病性及机械性溶血。',
        'on_jaundice_en': [
            'Confirm haemolysis: low haptoglobin, raised LDH/reticulocytes, positive DAT (Coombs).',
            'Avoid oxidative triggers in G6PD deficiency; avoid cold exposure in cold-AIHA.',
            'Treat the underlying cause (steroids/rituximab for AIHA, transfusion support if severe).',
            'Monitor for renal injury and haemolytic crisis.',
        ],
        'on_jaundice_cn': [
            '证实溶血：结合珠蛋白降低、LDH/网织红细胞升高、Coombs试验阳性。',
            'G6PD缺乏者避免氧化诱发因素；冷抗体型AIHA避免受寒。',
            '针对病因治疗（AIHA用糖皮质激素/利妥昔单抗；重症输血支持）。',
            '监测肾功能损伤与溶血危象。',
        ],
        'referral_en': 'To hepatology if mixed/uncertain pattern; to ICU for haemolytic '
                       'crisis with organ failure.',
        'referral_cn': '混合型或不确定 → 肝病科；溶血危象伴器官衰竭 → ICU。',
        'guidelines': [],
    },
    'oncology': {
        'name_en': 'Oncology',
        'name_cn': '肿瘤科',
        'focus_en': 'Malignant obstructive jaundice (pancreatic head, cholangiocarcinoma, '
                    'ampullary, HCC with bile duct invasion) and cancer-related hepatotoxicity.',
        'focus_cn': '恶性梗阻性黄疸（胰头癌、胆管癌、壶腹癌、肝癌侵及胆管）及肿瘤相关的肝毒性。',
        'on_jaundice_en': [
            'Stage with contrast CT/MRI + tumour markers (CA19-9, CEA, AFP); biopsy for tissue.',
            'Decide curative resection vs palliative biliary drainage (PTCD/ERCP stent).',
            'Assess liver reserve before chemoembolisation/systemic therapy (Child-Pugh, ALBI).',
            'Watch for cholangitis behind any stent occlusion.',
        ],
        'on_jaundice_cn': [
            '增强CT/MRI分期+肿瘤标志物（CA19-9、CEA、AFP）；活检取病理。',
            '决定根治性切除或姑息性胆道引流（PTCD/ERCP支架）。',
            '化疗栓塞/系统治疗前评估肝储备（Child-Pugh、ALBI）。',
            '警惕支架堵塞继发胆管炎。',
        ],
        'referral_en': 'To HPB surgery / interventional radiology for drainage; to GI for ERCP; '
                       'to MDT for combined modality planning.',
        'referral_cn': '引流 → 肝胆外科/介入科；ERCP → 消化科；综合治疗 → MDT。',
        'guidelines': ['HCC_2024', 'EASL_HCC_2024', 'BCLC_2022', 'AASLD_HCC_2023', 'ACR_JAUNDICE'],
    },
    'emergency': {
        'name_en': 'Emergency Department',
        'name_cn': '急诊科',
        'focus_en': 'Rapid triage of new-onset jaundice: separate life-threatening causes '
                    '(ALF, septic cholangitis, haemolytic crisis) from stable outpatient work-up.',
        'focus_cn': '急性黄疸的快速分诊：区分危及生命的病因（ALF、化脓性胆管炎、溶血危象）'
                    '与可门诊评估者。',
        'on_jaundice_en': [
            'ABCDE + vitals; check glucose, INR, lactate, blood cultures if septic.',
            'Red flags → resuscitate now: altered mentation, hypotension, fever with rigors '
            '(Charcot triad / Reynolds pentad), INR>1.5, oliguria.',
            'Point-of-care ultrasound to look for biliary dilation / obstruction.',
            'Triage: ALF/cholangitis/haemolysis → admit/ICU; stable → outpatient referral.',
        ],
        'on_jaundice_cn': [
            'ABCDE+生命体征；查血糖、INR、乳酸，脓毒症时血培养。',
            '危险信号 → 立即复苏：意识改变、低血压、发热伴寒战（Charcot三联征/Reynolds五联征）、'
            'INR>1.5、少尿。',
            '床旁超声查找胆道扩张/梗阻。',
            '分诊：ALF/胆管炎/溶血 → 收治/ICU；稳定者 → 门诊转诊。',
        ],
        'referral_en': 'Distribute to infectious-hepatology, GI, HPB surgery, haematology '
                       'or ICU based on the working diagnosis.',
        'referral_cn': '按初步诊断分流至感染/肝病科、消化科、肝胆外科、血液科或ICU。',
        'guidelines': ['TOKYO_TG18', 'LF_2018', 'AASLD_ALF_2023', 'SURVIVING_SEPSIS_2021', 'ACR_JAUNDICE'],
    },
    'interventional': {
        'name_en': 'Interventional Radiology',
        'name_cn': '介入科',
        'focus_en': 'Image-guided biliary drainage (PTCD), liver-tumour intervention '
                    '(TACE/TAE/TAI), portal/Budd-Chiari interventions, and decompression '
                    'of obstructive jaundice.',
        'focus_cn': '影像引导下胆道引流（PTCD）、肝肿瘤介入治疗（TACE/TAE/TAI）、'
                    '门脉/布加综合征介入，以及梗阻性黄疸减压。',
        'on_jaundice_en': [
            'PTCD for malignant / refractory biliary obstruction when ERCP or surgical '
            'bypass is not feasible.',
            'Correct coagulopathy (INR<1.5, platelets) before puncture; assess bleeding risk.',
            'For HCC: stage by CNLC and assess liver reserve (Child-Pugh/ALBI) before TACE/ablation.',
            'Watch for stent occlusion / cholangitis, post-procedure bleeding, bile leak.',
        ],
        'on_jaundice_cn': [
            'PTCD 用于恶性/难治性胆道梗阻减压（无法ERCP或外科旁路时）。',
            '穿刺前纠正凝血障碍（INR<1.5、血小板），评估出血风险。',
            'HCC 按CNLC分期并评估肝储备（Child-Pugh/ALBI）后决定TACE/消融。',
            '警惕支架堵塞/胆管炎、术后出血、胆汁漏。',
        ],
        'referral_en': 'Joint decision with oncology / HPB surgery; hepatology for liver '
                       'function; ICU for post-procedure liver failure or sepsis.',
        'referral_cn': '与肿瘤科/肝胆外科联合决策；肝病科评估肝功能；术后肝衰或脓毒症入ICU。',
        'guidelines': ['HCC_2024', 'ACUTE_BILIARY_2021'],
    },
    'liver_transplant': {
        'name_en': 'Liver Transplant',
        'name_cn': '肝移植科',
        'focus_en': 'Transplant evaluation and peri-operative care for end-stage liver '
                    'disease, ALF and ACLF.',
        'focus_cn': '终末期肝病、ALF/ACLF 的肝移植评估与围术期管理。',
        'on_jaundice_en': [
            'Stratify transplant priority by MELD-Na / MELD 3.0; grade decompensation by Child-Pugh.',
            'ALF meeting King\'s College Criteria / high-grade ACLF → urgent transplant evaluation.',
            'Screen contraindications: uncontrolled infection, extra-hepatic malignancy, '
            'adherence, psychosocial support.',
            'Peri-operative control of infection, renal function, hepatic encephalopathy.',
        ],
        'on_jaundice_cn': [
            '用 MELD-Na / MELD 3.0 评估移植优先级；用 Child-Pugh 评估失代偿程度。',
            'ALF 符合 King 学院标准 / ACLF 高分级 → 紧急肝移植评估。',
            '筛查禁忌：未控感染、肝外恶性肿瘤、依从性、社会心理支持。',
            '围术期控制感染、肾功能、肝性脑病。',
        ],
        'referral_en': 'ICU for organ support; infectious-hepatology for aetiology; '
                       'oncology to exclude malignancy.',
        'referral_cn': 'ICU 器官支持；感染/肝病科病因治疗；肿瘤科排除恶性。',
        'guidelines': ['LF_2018', 'APASL_ACLF_2019', 'KINGS_COLLEGE_ALF', 'MELD_3_0',
                       'MELD_ORIGINAL_2001', 'MELD_NA_2006', 'MELD_3_0_2021', 'CHILD_PUGH',
                       'CHINA_LIVER_TRANSPLANT_2018', 'CHINA_CLTR_MELD'],
    },
    'rheumatology': {
        'name_en': 'Rheumatology / Clinical Immunology',
        'name_cn': '风湿免疫科',
        'focus_en': 'Systemic evaluation and immunomodulation of autoimmune liver disease '
                    '(AIH / PBC / PSC) and overlap syndromes.',
        'focus_cn': '自身免疫性肝病（AIH/PBC/PSC）的系统评估与免疫调节；重叠综合征。',
        'on_jaundice_en': [
            'AIH: ANA / ASMA / anti-LKM, high IgG; steroids ± azathioprine.',
            'PBC: AMA; ursodeoxycholic acid (UDCA) first-line; obeticholic acid / fibrates if incomplete response.',
            'PSC: cholangiography; screen for IBD and cholangiocarcinoma.',
            'Watch for overlap syndromes (AIH-PBC / AIH-PSC) and immunosuppression side-effects.',
        ],
        'on_jaundice_cn': [
            'AIH：ANA/ASMA/抗LKM、IgG升高；糖皮质激素±硫唑嘌呤。',
            'PBC：AMA阳性；熊去氧胆酸(UDCA)一线；应答不全加奥贝胆酸/贝特类。',
            'PSC：胆管造影；筛查IBD与胆管癌。',
            '注意重叠综合征（AIH-PBC / AIH-PSC）及免疫抑制副作用。',
        ],
        'referral_en': 'Joint care with hepatology / GI; transplant for end-stage disease.',
        'referral_cn': '与肝病科/消化科协同；终末期考虑肝移植。',
        'guidelines': ['AIH_2021', 'PBC_2021', 'PSC_2023'],
    },
    'geriatrics': {
        'name_en': 'Geriatrics',
        'name_cn': '老年科',
        'focus_en': 'Jaundice assessment in elderly patients with multimorbidity; high '
                    'prevalence of drug-induced liver injury (polypharmacy); frailty and nutrition.',
        'focus_cn': '老年多病共存患者的黄疸评估；药物性肝损伤（多重用药）高发；衰弱与营养。',
        'on_jaundice_en': [
            'Systematic medication review (polypharmacy); watch for DILI (Hy\'s law).',
            'Assess frailty, nutrition and comorbidities before invasive steps.',
            'Imaging is first-line (non-invasive); use invasive procedures cautiously (bleeding/infection risk).',
            'Beware atypical presentation (infection, malignancy may be occult).',
        ],
        'on_jaundice_cn': [
            '系统审查用药（多重用药），警惕 DILI（Hy 定律）。',
            '有创操作前评估衰弱、营养与合并症。',
            '影像首选（无创），谨慎有创操作（出血/感染风险）。',
            '警惕不典型表现（感染、肿瘤常隐匿）。',
        ],
        'referral_en': 'Hepatology / GI for aetiology; surgery if obstruction; comprehensive '
                       'geriatric assessment.',
        'referral_cn': '肝病科/消化科查因；梗阻转外科；老年综合评估。',
        'guidelines': ['DILI_2023', 'CIRRHOSIS_2019', 'EASL_DILI_2019'],
    },
    'obstetrics': {
        'name_en': 'Obstetrics / Maternal-Fetal Medicine',
        'name_cn': '产科 / 母胎医学',
        'focus_en': 'Pregnancy-related liver disease where jaundice TYPE critically changes '
                    'management: intrahepatic cholestasis of pregnancy (ICP), HELLP syndrome '
                    'and acute fatty liver of pregnancy (AFLP).',
        'focus_cn': '妊娠相关肝病，黄疸类型决定截然不同的处理：妊娠期肝内胆汁淤积症(ICP)、'
                    'HELLP综合征、妊娠期急性脂肪肝(AFLP)。',
        'on_jaundice_en': [
            'Distinguish pregnancy-specific causes (ICP/HELLP/AFLP) from coincidental liver disease.',
            'ICP (cholestatic, pruritus, high bile acids): UDCA; fetal monitoring; timed delivery.',
            'HELLP/AFLP (hepatocellular, often third trimester): urgent evaluation, delivery, ICU.',
            'Check bile acids, LFTs, INR, platelets, glucose, ammonia (AFLP hypoglycaemia).',
        ],
        'on_jaundice_cn': [
            '鉴别妊娠特发性肝病（ICP/HELLP/AFLP）与合并的其他肝病。',
            'ICP（胆汁郁积、瘙痒、胆酸高）：熊去氧胆酸(UDCA)；胎心监测；适时终止妊娠。',
            'HELLP/AFLP（肝细胞型，常孕晚期）：紧急评估、终止妊娠、ICU支持。',
            '查胆酸、肝功能、INR、血小板、血糖、血氨（AFLP易低血糖）。',
        ],
        'referral_en': 'Maternal-fetal medicine + hepatology + ICU for HELLP/AFLP; neonatal '
                       'team for haemolytic disease of the newborn.',
        'referral_cn': '母胎医学+肝病科+ICU处理HELLP/AFLP；新生儿科防治新生儿溶血病。',
        'guidelines': ['RCOG_ICP_2023', 'ACOG_ICP_2023', 'ICP_CHINA_2024',
                       'ACOG_PREECLAMPSIA_2020', 'CHINA_HDP_2020'],
    },
    'nephrology': {
        'name_en': 'Nephrology',
        'name_cn': '肾内科',
        'focus_en': 'Jaundice overlapping with kidney injury where TYPE drives treatment: '
                    'hepatorenal syndrome (HRS-AKI), haemolytic-uraemic syndrome (HUS/TMA), '
                    'and post-obstructive AKI.',
        'focus_cn': '黄疸合并肾损伤，类型决定治疗：肝肾综合征(HRS-AKI)、溶血尿毒综合征(HUS/TMA)、'
                    '梗阻后AKI。',
        'on_jaundice_en': [
            'Hepatorenal syndrome (HRS-AKI in cirrhosis): albumin + terlipressin, transplant evaluation.',
            'HUS/TMA (haemolysis + AKI + thrombocytopenia): plasma exchange, complement work-up, dialysis.',
            'Post-obstructive AKI: relieve biliary obstruction; manage electrolytes and volume.',
            'Avoid nephrotoxins in jaundiced patients; dose-adjust all drugs.',
        ],
        'on_jaundice_cn': [
            '肝肾综合征（肝硬化HRS-AKI）：白蛋白+特利加压素、评估移植（ICA共识）。',
            'HUS/TMA（溶血+AKI+血小板减少）：血浆置换、补体评估、透析。',
            '梗阻后AKI：解除胆道梗阻，管理电解质与容量。',
            '黄疸患者避免肾毒性药物，所有药物按肾功能调整剂量。',
        ],
        'referral_en': 'Hepatology/GI for HRS; haematology for TMA/HUS; HPB surgery for obstruction.',
        'referral_cn': 'HRS→肝病/消化科；TMA/HUS→血液科；梗阻→肝胆外科。',
        'guidelines': ['KDIGO_AKI_2012', 'ICA_HRS_2024', 'TMA_HUS_GUIDANCE'],
    },
    'stomatology': {
        'name_en': 'Stomatology (Dentistry)',
        'name_cn': '口腔科',
        'focus_en': 'Incidental detection of oral-mucosal yellowing; bleeding and infection '
                    'risk of oral procedures in liver-disease patients (screening sentinel).',
        'focus_cn': '口腔黏膜/舌下黄染的偶然发现；肝病患者口腔操作的出血与感染风险（筛查前哨）。',
        'on_jaundice_en': [
            'Oral-mucosa / hard-palate yellowing suggests jaundice → check serum bilirubin.',
            'Before oral procedures in liver-disease patients, assess INR / platelets.',
            'Oral signs may be the first presentation of liver disease (fetor hepaticus, bleeding tendency).',
        ],
        'on_jaundice_cn': [
            '口腔黏膜/硬腭黄染提示黄疸 → 建议查血清胆红素。',
            '肝病/凝血异常患者口腔操作前评估 INR/血小板。',
            '警惕口腔表现可能是肝病首发（肝臭、出血倾向）。',
        ],
        'referral_en': 'Refer to infectious-hepatology or GI for aetiology.',
        'referral_cn': '转诊感染/肝病科或消化科查因。',
        'guidelines': [],
    },
    'ophthalmology': {
        'name_en': 'Ophthalmology',
        'name_cn': '眼科',
        'focus_en': 'Scleral icterus is the earliest and most reliable sign of jaundice; '
                    'often first noticed here (screening sentinel).',
        'focus_cn': '巩膜黄染是黄疸最早、最可靠的体征，眼科常首先发现（筛查前哨）。',
        'on_jaundice_en': [
            'Scleral icterus (best in natural light) → advise prompt serum bilirubin + liver function.',
            'Differentiate from pinguecula / pterygium / fat deposits (asymmetric, movable).',
            'Note accompanying pruritus (cholestasis).',
        ],
        'on_jaundice_cn': [
            '巩膜黄染（自然光下最佳）→ 建议尽快查血清胆红素与肝功能。',
            '与睑裂斑/翼状胬肉/脂肪沉着鉴别（不对称、可移动）。',
            '记录是否伴皮肤瘙痒（胆汁淤积）。',
        ],
        'referral_en': 'Refer to GI / hepatology.',
        'referral_cn': '转诊消化/肝病科。',
        'guidelines': [],
    },
    'public_health': {
        'name_en': 'Public Health / Health Check-up',
        'name_cn': '公卫 / 体检 / 预防保健',
        'focus_en': 'Population screening and detection of jaundice at check-up; public-health '
                    'significance of viral hepatitis (screening sentinel).',
        'focus_cn': '人群筛查与体检中黄疸的发现；病毒性肝炎的公卫意义（筛查前哨）。',
        'on_jaundice_en': [
            'Skin/scleral yellowing at check-up → serum bilirubin + liver function + viral serology.',
            'Report and follow up viral hepatitis per public-health requirements.',
            'Health education: abstinence, safe medication, vaccination.',
        ],
        'on_jaundice_cn': [
            '体检发现皮肤/巩膜黄染 → 血清胆红素 + 肝功能 + 病毒血清学。',
            '病毒性肝炎按公卫要求上报与随访。',
            '健康宣教：戒酒、安全用药、疫苗接种。',
        ],
        'referral_en': 'Refer positive cases to specialist care.',
        'referral_cn': '阳性者转诊专科。',
        'guidelines': ['CHB_2022'],
    },
    'other': {
        'name_en': 'Other Departments',
        'name_cn': '其他科室',
        'focus_en': 'Non-hepato-biliary / haematology specialties (dermatology, endocrinology, '
                    'nephrology, cardiology, respiratory, neurology, general practice, TCM, '
                    'rehabilitation, etc.) encountering jaundice incidentally. Jaundice is '
                    'usually a manifestation of hepato-biliary or haematological disease — '
                    'specialist referral and basic work-up are the priority.',
        'focus_cn': '非肝病/胆道/血液专科的科室（皮肤、内分泌、肾内、心内、呼吸、神经、'
                    '全科、中医、康复等）偶见黄疸。黄疸多为肝病/胆道/血液系统疾病的表现，'
                    '应以专科转诊与基础筛查为优先。',
        'on_jaundice_en': [
            'Basic screening first: liver function, bilirubin, CBC, abdominal ultrasound.',
            'Suspect drug-induced liver injury related to this department\'s medications.',
            'Jaundice is rarely a primary problem here — refer to the relevant specialty '
            '(hepatology / GI / HPB surgery / haematology).',
        ],
        'on_jaundice_cn': [
            '先做基础筛查：肝功能、胆红素、血常规、腹部超声。',
            '警惕本科用药相关的药物性肝损伤。',
            '黄疸在本科多为继发表现 → 转诊相应专科（肝病/消化/肝胆外科/血液科）。',
        ],
        'referral_en': 'Refer to infectious-hepatology, GI, HPB surgery or haematology as appropriate.',
        'referral_cn': '按情况转诊感染/肝病科、消化科、肝胆外科或血液科。',
        'guidelines': [],
    },
}


# ═════════════════════════════════════════════════════════════
# 1b. DEPARTMENT × JAUNDICE-TYPE DECISION MATRIX
# ═════════════════════════════════════════════════════════════
# For each department, how the JAUNDICE TYPE changes the clinical decision.
# This is the "依据黄疸类型导致不同临床决策" knowledge: hepatocellular / cholestatic /
# hemolytic each trigger a different action in the same department.

DEPARTMENT_TYPE_DECISIONS = {
    'infectious_hepatology': {
        'cn': {
            'hepatocellular': '病因筛查（病毒/DILI/酒精/自身免疫）；按指征抗病毒（HBV核苷类似物）、停药、免疫抑制；评估肝衰竭（COSSH/CLIF-C）。',
            'cholestatic': '首选MRCP排除肝外梗阻；肝内淤积查PBC(AMA)/PSC/药物性；UDCA对症退黄。',
            'hemolytic': '非本科重点 → 转血液科（Coombs/网织红细胞/结合珠蛋白）。',
        },
        'en': {
            'hepatocellular': 'Aetiology work-up (viral/DILI/alcohol/autoimmune); aetiology-specific therapy (HBV NAs, drug withdrawal, immunosuppression); assess liver failure (COSSH/CLIF-C).',
            'cholestatic': 'MRCP first to exclude extra-hepatic obstruction; for intra-hepatic cholestasis test PBC(AMA)/PSC/drug-related; UDCA for symptom relief.',
            'hemolytic': 'Not this department\'s focus → refer to haematology (Coombs/reticulocytes/haptoglobin).',
        },
    },
    'gi': {
        'cn': {
            'hepatocellular': '肝病病因筛查+纤维化/肝硬化评估；失代偿者筛查食管胃静脉曲张。',
            'cholestatic': 'MRCP→ERCP（取石/支架/狭窄扩张）；查PBC/PSC；EUS评估胰头病变。',
            'hemolytic': '转血液科。',
        },
        'en': {
            'hepatocellular': 'Liver aetiology work-up + fibrosis/cirrhosis assessment; variceal screening if decompensated.',
            'cholestatic': 'MRCP→ERCP (stone removal/stent/stricture); test PBC/PSC; EUS for pancreatic head lesions.',
            'hemolytic': 'Refer to haematology.',
        },
    },
    'hpb_surgery': {
        'cn': {
            'hepatocellular': '通常非外科指征 → 转内科（肝病/感染科）。',
            'cholestatic': '外科或内镜处理（胆总管取石/胆道引流/肿瘤切除）；术前纠正凝血（INR<1.5、血小板）。',
            'hemolytic': '转血液科（除非溶血继发胆石需手术）。',
        },
        'en': {
            'hepatocellular': 'Usually NOT a surgical indication → refer to medicine (hepatology/infectious disease).',
            'cholestatic': 'Surgical or endoscopic management (CBD stone removal/biliary drainage/tumour resection); correct coagulopathy pre-op (INR<1.5, platelets).',
            'hemolytic': 'Refer to haematology (unless pigment stones from haemolysis need surgery).',
        },
    },
    'icu': {
        'cn': {
            'hepatocellular': 'ALF/ACLF集束化（不论病因NAC、颅内压监测、移植评估；King\'s/COSSH评分）。',
            'cholestatic': '胆道引流控制感染源；脓毒症复苏（Surviving Sepsis）+抗生素。',
            'hemolytic': '溶血危象支持（停诱因、输血、激素）；肾替代治疗。',
        },
        'en': {
            'hepatocellular': 'ALF/ACLF bundle (NAC regardless of cause, ICP monitoring, transplant evaluation; King\'s/COSSH scoring).',
            'cholestatic': 'Biliary drainage for source control; sepsis resuscitation (Surviving Sepsis) + antibiotics.',
            'hemolytic': 'Haemolytic crisis support (stop trigger, transfusion, steroids); renal replacement.',
        },
    },
    'hematology': {
        'cn': {
            'hepatocellular': '转肝病/消化科。',
            'cholestatic': '多为胆石相关 → 转消化/外科。',
            'hemolytic': '病因治疗（AIHA糖皮质激素/利妥昔单抗、G6PD避诱因、遗传性溶血输血支持）；监测肾与溶血危象。',
        },
        'en': {
            'hepatocellular': 'Refer to hepatology/GI.',
            'cholestatic': 'Usually gallstone-related → refer to GI/surgery.',
            'hemolytic': 'Cause-specific treatment (AIHA steroids/rituximab, G6PD trigger avoidance, transfusion for hereditary haemolysis); monitor kidney and crisis.',
        },
    },
    'oncology': {
        'cn': {
            'hepatocellular': 'HCC按BCLC/CNLC分期（切除/消融/TACE/移植/系统治疗）；评估肝储备（Child-Pugh/ALBI）。',
            'cholestatic': '恶性梗阻引流（PTCD/ERCP支架）+肿瘤分期+MDT决策。',
            'hemolytic': '瘤相关或治疗相关溶血 → 与血液科协同。',
        },
        'en': {
            'hepatocellular': 'Stage HCC by BCLC/CNLC (resection/ablation/TACE/transplant/systemic); assess liver reserve (Child-Pugh/ALBI).',
            'cholestatic': 'Malignant obstruction drainage (PTCD/ERCP stent) + staging + MDT decision.',
            'hemolytic': 'Tumour- or treatment-related haemolysis → coordinate with haematology.',
        },
    },
    'emergency': {
        'cn': {
            'hepatocellular': '识别ALF（INR>1.5+脑病）→ ICU/紧急移植评估。',
            'cholestatic': 'Charcot三联征/Reynolds五联征 → 胆道急诊+血培养+抗生素。',
            'hemolytic': '溶血危象 → 复苏+紧急血液科会诊。',
        },
        'en': {
            'hepatocellular': 'Recognise ALF (INR>1.5 + encephalopathy) → ICU/urgent transplant evaluation.',
            'cholestatic': 'Charcot triad / Reynolds pentad → biliary emergency + blood cultures + antibiotics.',
            'hemolytic': 'Haemolytic crisis → resuscitation + urgent haematology consult.',
        },
    },
    'interventional': {
        'cn': {
            'hepatocellular': 'HCC的TACE/消融（评估肝储备Child-Pugh/ALBI）；布加综合征血管介入。',
            'cholestatic': 'PTCD胆道减压（恶性或难治性梗阻，无法ERCP/外科旁路时）。',
            'hemolytic': '通常非介入指征 → 转血液科。',
        },
        'en': {
            'hepatocellular': 'TACE/ablation for HCC (assess liver reserve Child-Pugh/ALBI); vascular intervention for Budd-Chiari.',
            'cholestatic': 'PTCD biliary decompression (malignant or refractory obstruction, when ERCP/surgical bypass not feasible).',
            'hemolytic': 'Usually not an interventional indication → refer to haematology.',
        },
    },
    'liver_transplant': {
        'cn': {
            'hepatocellular': 'ALF/ACLF/终末期肝硬化移植评估（MELD 3.0/King\'s学院标准）。',
            'cholestatic': 'PBC/PSC终末期可考虑移植；恶性梗阻通常为移植禁忌。',
            'hemolytic': '遗传性溶血致继发性胆汁性肝硬化/胆石终末期可评估移植。',
        },
        'en': {
            'hepatocellular': 'Transplant evaluation for ALF/ACLF/end-stage cirrhosis (MELD 3.0 / King\'s College Criteria).',
            'cholestatic': 'End-stage PBC/PSC may be considered; malignant obstruction is usually a contraindication.',
            'hemolytic': 'Secondary biliary cirrhosis / end-stage from hereditary haemolysis may be evaluated.',
        },
    },
    'rheumatology': {
        'cn': {
            'hepatocellular': '自身免疫性肝炎AIH（ANA/ASMA/抗LKM、IgG升高；糖皮质激素±硫唑嘌呤）。',
            'cholestatic': 'PBC（AMA阳性、UDCA一线）；PSC（胆管造影、筛查IBD与胆管癌）。',
            'hemolytic': '继发于自身免疫病的AIHA → 免疫抑制治疗。',
        },
        'en': {
            'hepatocellular': 'Autoimmune hepatitis AIH (ANA/ASMA/anti-LKM, high IgG; steroids ± azathioprine).',
            'cholestatic': 'PBC (AMA-positive, UDCA first-line); PSC (cholangiography, screen IBD and cholangiocarcinoma).',
            'hemolytic': 'AIHA secondary to autoimmune disease → immunosuppression.',
        },
    },
    'geriatrics': {
        'cn': {
            'hepatocellular': '多重用药DILI筛查（Hy定律）；衰弱与合并症评估。',
            'cholestatic': '影像首选（无创）；谨慎有创操作（出血/感染风险）。',
            'hemolytic': '警惕药物或机械瓣相关溶血。',
        },
        'en': {
            'hepatocellular': 'Polypharmacy DILI screen (Hy\'s law); frailty and comorbidity assessment.',
            'cholestatic': 'Imaging first-line (non-invasive); cautious invasive procedures (bleeding/infection risk).',
            'hemolytic': 'Beware drug- or prosthetic-valve-related haemolysis.',
        },
    },
    'obstetrics': {
        'cn': {
            'hepatocellular': 'HELLP综合征 / 妊娠期急性脂肪肝(AFLP) → 紧急评估、考虑终止妊娠；母胎医学+ICU协同。',
            'cholestatic': '妊娠期肝内胆汁淤积症(ICP) → 熊去氧胆酸(UDCA)、胎心监测、按孕周与胆酸决定分娩时机。',
            'hemolytic': 'HELLP伴微血管病性溶血；母胎血型不合(ABO/Rh) → 抗D预防、新生儿溶血病防治。',
        },
        'en': {
            'hepatocellular': 'HELLP syndrome / acute fatty liver of pregnancy (AFLP) → urgent evaluation, consider delivery; maternal-fetal medicine + ICU.',
            'cholestatic': 'Intrahepatic cholestasis of pregnancy (ICP) → UDCA, fetal monitoring, timed delivery by gestational age and bile acids.',
            'hemolytic': 'Microangiopathic haemolysis with HELLP; ABO/Rh alloimmunisation → anti-D prophylaxis, haemolytic disease of the newborn prevention.',
        },
    },
    'nephrology': {
        'cn': {
            'hepatocellular': '肝肾综合征HRS-AKI（白蛋白+特利加压素、评估移植；ICA共识）。',
            'cholestatic': '梗阻/胆道术后AKI → 解除梗阻、管理电解质与容量。',
            'hemolytic': '溶血尿毒综合征HUS/TMA → 血浆置换、补体评估、透析。',
        },
        'en': {
            'hepatocellular': 'Hepatorenal syndrome HRS-AKI (albumin + terlipressin, transplant evaluation; ICA consensus).',
            'cholestatic': 'Post-obstructive / post-biliary-surgery AKI → relieve obstruction, manage electrolytes and volume.',
            'hemolytic': 'Haemolytic-uraemic syndrome HUS/TMA → plasma exchange, complement work-up, dialysis.',
        },
    },
    'stomatology': {
        'cn': {
            'hepatocellular': '提示肝病 → 查肝功能与凝血（口腔操作出血风险），转肝病/感染科。',
            'cholestatic': '提示胆道问题 → 转消化科。',
            'hemolytic': '提示溶血 → 转血液科。',
        },
        'en': {
            'hepatocellular': 'Suggests liver disease → check LFTs and coagulation (oral-procedure bleeding risk), refer to hepatology.',
            'cholestatic': 'Suggests biliary disease → refer to GI.',
            'hemolytic': 'Suggests haemolysis → refer to haematology.',
        },
    },
    'ophthalmology': {
        'cn': {
            'hepatocellular': '巩膜黄染 → 查胆红素+肝功能 → 转肝病/感染科查因。',
            'cholestatic': '巩膜黄染+瘙痒 → 查胆红素/ALP → 转消化科。',
            'hemolytic': '巩膜黄染（间接胆红素）→ 转血液科。',
        },
        'en': {
            'hepatocellular': 'Scleral icterus → check bilirubin+LFTs → refer to hepatology.',
            'cholestatic': 'Scleral icterus + pruritus → check bilirubin/ALP → refer to GI.',
            'hemolytic': 'Scleral icterus (indirect bilirubin) → refer to haematology.',
        },
    },
    'public_health': {
        'cn': {
            'hepatocellular': '体检黄染 → 查肝功能/病毒血清学 → 阳性转肝病/感染科。',
            'cholestatic': '体检黄染 → 超声+MRCP → 转消化/外科。',
            'hemolytic': '体检黄染（间接胆红素）→ 转血液科。',
        },
        'en': {
            'hepatocellular': 'Check-up yellowing → LFTs/viral serology → refer to hepatology if positive.',
            'cholestatic': 'Check-up yellowing → ultrasound+MRCP → refer to GI/surgery.',
            'hemolytic': 'Check-up yellowing (indirect bilirubin) → refer to haematology.',
        },
    },
    'other': {
        'cn': {
            'hepatocellular': '基础筛查（肝功能/超声）→ 转肝病/感染/消化科。',
            'cholestatic': '基础筛查 → 转消化/肝胆外科。',
            'hemolytic': '基础筛查（血常规/Coombs）→ 转血液科。',
        },
        'en': {
            'hepatocellular': 'Basic work-up (LFTs/ultrasound) → refer to hepatology/infectious/GI.',
            'cholestatic': 'Basic work-up → refer to GI/HPB surgery.',
            'hemolytic': 'Basic work-up (CBC/Coombs) → refer to haematology.',
        },
    },
}


# ═════════════════════════════════════════════════════════════
# 2. DISEASES  —  profiles inferred from the admission diagnosis
# ═════════════════════════════════════════════════════════════
# Keywords are matched (case-insensitive substring) against the diagnosis free text.
# ``priority`` controls ordering: higher = checked first (most acute/specific).

DISEASES = {
    'alf': {
        'name_en': 'Acute Liver Failure (ALF)',
        'name_cn': '急性肝衰竭',
        'keywords': ['急性肝衰竭', '急性肝功能衰竭', '暴发性', 'fulminant'],
        # exclude chronic-on-acute / subacute variants (handled by their own entries)
        'exclude': ['慢加', '亚急'],
        'priority': 100,
        'jaundice_type': 'hepatocellular',
        'department': 'icu',
        'pathway': 'acute_liver_failure',
        'guidelines': ['LF_2018', 'KINGS_COLLEGE_ALF', 'AASLD_ALF_2023', 'EASL_COAG_2022'],
        'principle_en': 'Time-critical: NAC regardless of cause, intracranial-pressure '
                        'vigilance, identify and remove precipitant, transplant evaluation.',
        'principle_cn': '争分夺秒：不论病因给予NAC，警惕颅内压，识别并去除诱因，评估肝移植指征。',
    },
    'salf': {
        'name_en': 'Subacute / Late-onset Liver Failure',
        'name_cn': '亚急性肝衰竭',
        'keywords': ['亚急性肝衰竭', '慢加亚急性肝衰竭'],
        'priority': 102,
        'jaundice_type': 'hepatocellular',
        'department': 'infectious_hepatology',
        'pathway': 'acute_liver_failure',
        'guidelines': ['LF_2018', 'KINGS_COLLEGE_ALF'],
        'principle_en': 'Subacute course (2-26 weeks); intensive medical care, treat the '
                        'underlying cause, transplant evaluation if prothrombin activity < 40%.',
        'principle_cn': '病程亚急性（2-26周）；强化内科治疗，针对病因处理，凝血酶原活动度<40%时评估移植。',
    },
    'aclf': {
        'name_en': 'Acute-on-Chronic Liver Failure (ACLF)',
        'name_cn': '慢加急性肝衰竭',
        'keywords': ['慢加急性肝衰竭', '慢加亚急性', 'aclf'],
        'priority': 103,
        'jaundice_type': 'hepatocellular',
        'department': 'infectious_hepatology',
        'pathway': 'aclf',
        'guidelines': ['LF_2018', 'APASL_ACLF_2019', 'EASL_CLIF_2013', 'EASL_DECOMP_2018', 'SURVIVING_SEPSIS_2021',
                       'CHILD_PUGH', 'MELD_3_0', 'MELD_ORIGINAL_2001', 'MELD_NA_2006', 'MELD_3_0_2021'],
        'principle_en': 'Identify and treat the precipitant (infection, bleeding, alcohol, '
                        'hepatotoxic drugs, HBV reactivation); score with COSSH-ACLF / '
                        'CLIF-C ACLF; organ support; early NAs for HBV-ACLF.',
        'principle_cn': '识别并治疗诱因（感染、出血、酒精、肝毒性药物、HBV再激活）；'
                        '用 COSSH-ACLF / CLIF-C ACLF 评分；器官支持；HBV-ACLF 尽早启动核苷类似物。',
    },
    'malignant_obstruction': {
        'name_en': 'Malignant Biliary Obstruction',
        'name_cn': '恶性胆道梗阻（胰头癌/胆管癌/壶腹癌/肝门部）',
        'keywords': ['胰头癌', '胆管癌', '壶腹', '肝门部', 'pancreatic head',
                     'cholangiocarcinoma', 'klatskin'],
        'priority': 90,
        'jaundice_type': 'cholestatic',
        'department': 'oncology',
        'pathway': 'obstructive_jaundice',
        'guidelines': ['HCC_2024', 'ACUTE_BILIARY_2021', 'ACR_JAUNDICE', 'BCLC_2022'],
        'principle_en': 'Painless progressive jaundice; stage and plan; biliary drainage '
                        '(PTCD / ERCP stent) for decompression, assess resectability.',
        'principle_cn': '无痛性进行性黄疸；分期并制定方案；胆道引流（PTCD / ERCP支架）减压，'
                        '评估可切除性。',
    },
    'hcc': {
        'name_en': 'Hepatocellular Carcinoma (HCC)',
        'name_cn': '肝细胞癌 / 肝恶性肿瘤',
        'keywords': ['肝恶性肿瘤', '肝癌', '肝细胞癌', 'hcc', '肝右叶占位', '肝左叶占位', '肝占位'],
        'priority': 88,
        'jaundice_type': 'mixed',
        'department': 'oncology',
        'pathway': None,
        'guidelines': ['HCC_2024', 'CIRRHOSIS_2019', 'EASL_HCC_2024', 'BCLC_2022', 'AASLD_HCC_2023',
                       'ALBI_2015', 'CHILD_PUGH', 'MELD_3_0'],
        'principle_en': 'Stage by CNLC/BCLC; assess liver reserve (Child-Pugh/ALBI); '
                        'curative (resection/ablation/transplant) vs palliative (TACE/systemic).',
        'principle_cn': '按CNLC/BCLC分期；评估肝储备（Child-Pugh/ALBI）；'
                        '根治性（切除/消融/移植）vs 姑息性（TACE/系统治疗）。',
    },
    'cholangitis': {
        'name_en': 'Acute Cholangitis / Choledocholithiasis',
        'name_cn': '急性胆管炎 / 胆总管结石',
        'keywords': ['胆管炎', '胆总管结石', '胆道感染', 'cholangitis', 'choledocholithiasis'],
        # exclude PBC / PSC (胆管炎 is a substring of 胆汁性/淤积性胆管炎)
        'exclude': ['原发性胆汁性', '原发性胆汁淤积性', 'pbc', '硬化性胆管炎', 'psc'],
        'priority': 95,
        'jaundice_type': 'cholestatic',
        'department': 'hpb_surgery',
        'pathway': 'acute_cholangitis',
        'guidelines': ['ACUTE_BILIARY_2021', 'TOKYO_TG18'],
        'principle_en': 'Grade severity by TG18; antibiotics first, then biliary drainage '
                        '(timing depends on grade); ERCP is first-line for stones.',
        'principle_cn': '按TG18分级严重程度；先抗生素，再胆道引流（时机依分级而定）；'
                        '结石首选ERCP。',
    },
    'choledocholithiasis': {
        'name_en': 'Choledocholithiasis / Gallstone Disease',
        'name_cn': '胆总管结石 / 胆石症',
        'keywords': ['胆石症', '胆囊结石', '胆总管', 'gallstone', 'choledoch'],
        'priority': 70,
        'jaundice_type': 'cholestatic',
        'department': 'hpb_surgery',
        'pathway': 'obstructive_jaundice',
        'guidelines': ['WORLD_GALLSTONE_2022', 'ACUTE_BILIARY_2021', 'ACR_JAUNDICE'],
        'principle_en': 'Distinguish asymptomatic stones from obstruction/cholangitis; '
                        'ERCP + laparoscopic cholecystectomy when indicated.',
        'principle_cn': '区分无症状结石与梗阻/胆管炎；指征明确时ERCP+腹腔镜胆囊切除术。',
    },
    'hemolysis': {
        'name_en': 'Haemolytic Jaundice',
        'name_cn': '溶血性黄疸 / 溶血性贫血',
        'keywords': ['溶血', '球形红细胞', '地中海贫血', 'g6pd', 'aiha', 'haemolys', 'hemolys',
                     '镰状', 'thalass', 'spherocyt'],
        'priority': 85,
        'jaundice_type': 'hemolytic',
        'department': 'hematology',
        'pathway': 'hemolytic_crisis',
        'guidelines': ['TMA_HUS_GUIDANCE'],
        'principle_en': 'Confirm haemolysis (low haptoglobin, high LDH/reticulocytes, DAT); '
                        'treat the cause; avoid oxidative triggers in G6PD deficiency.',
        'principle_cn': '证实溶血（结合珠蛋白低、LDH/网织红细胞高、Coombs试验）；'
                        '针对病因治疗；G6PD缺乏者避免氧化诱发因素。',
    },
    'psc': {
        'name_en': 'Primary Sclerosing Cholangitis (PSC)',
        'name_cn': '原发性硬化性胆管炎',
        'keywords': ['硬化性胆管炎', 'psc'],
        'priority': 80,
        'jaundice_type': 'cholestatic',
        'department': 'gi',
        'pathway': None,
        'guidelines': ['PSC_2023', 'AASLD_PSC_2024', 'EASL_PSC_2022'],
        'principle_en': 'Cholestatic pattern with biliary strictures; screen for IBD and '
                        'cholangiocarcinoma; transplant for end-stage disease.',
        'principle_cn': '胆汁郁积型伴胆管狭窄；筛查IBD与胆管癌；终末期考虑肝移植。',
    },
    'pbc': {
        'name_en': 'Primary Biliary Cholangitis (PBC)',
        'name_cn': '原发性胆汁性胆管炎',
        'keywords': ['原发性胆汁性胆管炎', '原发性胆汁淤积性', 'pbc'],
        'priority': 80,
        'jaundice_type': 'cholestatic',
        'department': 'infectious_hepatology',
        'pathway': None,
        'guidelines': ['PBC_2021', 'EASL_PBC_2017'],
        'principle_en': 'AMA-positive cholestatic disease; ursodeoxycholic acid (UDCA) '
                        'first-line; obeticholic acid / fibrates if incomplete response.',
        'principle_cn': 'AMA阳性胆汁郁积性疾病；熊去氧胆酸(UDCA)一线治疗；'
                        '应答不完全加用奥贝胆酸/贝特类。',
    },
    'aih': {
        'name_en': 'Autoimmune Hepatitis (AIH)',
        'name_name_cn': '自身免疫性肝炎',
        'name_cn': '自身免疫性肝炎',
        'keywords': ['自身免疫性肝炎', '自免肝', '自身免疫性肝病', 'aih'],
        'priority': 78,
        'jaundice_type': 'hepatocellular',
        'department': 'infectious_hepatology',
        'pathway': None,
        'guidelines': ['AIH_2021', 'EASL_AIH_2015', 'AASLD_AIH_2019'],
        'principle_en': 'ANA/ASMA/anti-LKM, high IgG; immunosuppression (steroids ± '
                        'azathioprine); distinguish from PBC/PSC overlap.',
        'principle_cn': 'ANA/ASMA/抗LKM阳性，IgG升高；免疫抑制（糖皮质激素±硫唑嘌呤）；'
                        '注意与PBC/PSC重叠综合征鉴别。',
    },
    'dili': {
        'name_en': 'Drug-induced Liver Injury (DILI)',
        'name_cn': '药物性肝损伤',
        'keywords': ['药物性肝损伤', '药物性肝', '药物性', 'dili', '药物性胆汁淤积'],
        'priority': 76,
        'jaundice_type': 'mixed',
        'department': 'infectious_hepatology',
        'pathway': 'dili_urgent',
        'guidelines': ['DILI_2023', 'EASL_DILI_2019'],
        'principle_en': 'Stop the offending drug; causality by RUCAM; supportive care; '
                        'NAC for APAP; watch for Hy\'s law (jaundice + ALT>3xULN → high mortality).',
        'principle_cn': '立即停用可疑药物；RUCAM因果评估；支持治疗；APAP用NAC；'
                        '警惕Hy定律（黄疸+ALT>3倍正常上限→死亡率高）。',
    },
    'alcoholic': {
        'name_en': 'Alcoholic Liver Disease / Alcoholic Hepatitis',
        'name_cn': '酒精性肝病 / 酒精性肝炎',
        'keywords': ['酒精性', 'alcoholic'],
        'priority': 74,
        'jaundice_type': 'hepatocellular',
        'department': 'infectious_hepatology',
        'pathway': None,
        'guidelines': ['ALD_2018', 'ACG_ALD_2018', 'EASL_ALD_2018', 'EASL_ARD_2024'],
        'principle_en': 'Abstinence is cornerstone; Maddrey score ≥32 → consider steroids '
                        '(if no contraindication); nutrition critical; Wernicke prophylaxis.',
        'principle_cn': '戒酒是基石；Maddrey评分≥32考虑糖皮质激素（无禁忌时）；'
                        '营养支持关键；预防Wernicke脑病。',
    },
    'decomp_cirrhosis': {
        'name_en': 'Decompensated Cirrhosis',
        'name_cn': '肝硬化失代偿期',
        'keywords': ['失代偿', '肝硬化', 'cirrhosis', '门脉高压', '腹水'],
        'priority': 60,
        'jaundice_type': 'hepatocellular',
        'department': 'infectious_hepatology',
        'pathway': None,
        'guidelines': ['CIRRHOSIS_2019', 'EASL_DECOMP_2018', 'CHILD_PUGH', 'MELD_3_0', 'EASL_COAG_2022',
                       'CHILD_TURCOTTE_1964', 'MELD_ORIGINAL_2001', 'MELD_NA_2006', 'MELD_3_0_2021',
                       'ALBI_2015', 'CHINA_LIVER_TRANSPLANT_2018', 'CHINA_PORTAL_HYPERTENSION_2023'],
        'principle_en': 'Assess Child-Pugh/MELD-Na; manage decompensation events '
                        '(ascites, SBP, encephalopathy, variceal bleeding); transplant referral.',
        'principle_cn': '评估Child-Pugh/MELD-Na；处理失代偿事件（腹水、自发性细菌性腹膜炎、'
                        '肝性脑病、曲张静脉出血）；肝移植转诊。',
    },
    'chronic_hbv': {
        'name_en': 'Chronic Viral Hepatitis (HBV / HCV)',
        'name_cn': '慢性病毒性肝炎（乙型/丙型，含相关肝硬化）',
        'keywords': ['乙型', '乙肝', 'hbv', '慢性乙型', '丙型', '丙肝', 'hcv', '慢性丙型'],
        'priority': 50,
        'jaundice_type': 'hepatocellular',
        'department': 'infectious_hepatology',
        'pathway': None,
        'guidelines': ['CHB_2022', 'CHC_2022', 'EASL_CHB_2017', 'AASLD_CHB_2024', 'HCV_GUIDANCE_2023'],
        'principle_en': 'Assess treatment indication: HBV by HBeAg/ALT/HBV-DNA/fibrosis '
                        '(nucleos(t)ide analogues entecavir/tenofovir first-line); HCV by '
                        'genotype and DAA regimen (curative). HCC surveillance in cirrhosis.',
        'principle_cn': '评估治疗指征：HBV按HBeAg/ALT/HBV-DNA/纤维化（核苷（酸）类似物'
                        '恩替卡韦/替诺福韦一线）；HCV按基因型选择DAA方案（可治愈）。'
                        '肝硬化者规律HCC监测。',
    },
    'hev': {
        'name_en': 'Hepatitis E (and other acute viral hepatitis)',
        'name_cn': '戊型病毒性肝炎（及其他急性病毒性肝炎）',
        'keywords': ['戊型', '甲型', 'hev', 'hav', '戊肝', '甲肝'],
        'priority': 50,
        'jaundice_type': 'hepatocellular',
        'department': 'infectious_hepatology',
        'pathway': None,
        'guidelines': ['CHB_2022', 'EASL_HEV_2018'],
        'principle_en': 'Mostly self-limited; supportive care; high risk of ALF in '
                        'pregnancy (HEV) and chronic liver disease; ribavirin in severe HEV.',
        'principle_cn': '多为自限性；支持治疗；妊娠期(HEV)及慢性肝病者易发生ALF；'
                        '重症HEV可用利巴韦林。',
    },
    'unexplained': {
        'name_en': 'Unexplained Liver Injury (aetiology unclear)',
        'name_cn': '肝损伤原因待查',
        'keywords': ['待查', '待诊', '原因不明', '异常原因', '原因待', 'unexplained'],
        'priority': 40,
        'jaundice_type': 'mixed',
        'department': 'infectious_hepatology',
        'pathway': None,
        'guidelines': ['CIRRHOSIS_2019', 'DILI_2023', 'AIH_2021', 'EASL_NIT_2021',
                       'EASL_WILSON_2012', 'EASL_VASCULAR_2016', 'ACR_JAUNDICE'],
        'principle_en': 'Systematic work-up: viral, autoimmune, metabolic, drug/alcohol, '
                        'vascular (Budd-Chiari), imaging; consider liver biopsy.',
        'principle_cn': '系统筛查：病毒、自身免疫、代谢、药物/酒精、血管（布加综合征）、影像；'
                        '必要时肝穿刺活检。',
    },
}


# ═════════════════════════════════════════════════════════════
# 3. JAUNDICE TYPES  (pathophysiological classification)
# ═════════════════════════════════════════════════════════════

JAUNDICE_TYPES = {
    'hepatocellular': {
        'en_name': 'Hepatocellular Jaundice',
        'cn_name': '肝细胞性黄疸',
        'mechanism': 'Liver cell damage impairs bilirubin uptake/conjugation.',
        'bilirubin_pattern': 'Mixed direct and indirect elevation.',
        'lab_pattern': 'ALT/AST >> ALP/GGT (ratio >2).',
        'common_causes_en': [
            'Viral hepatitis (HBV, HCV, HAV, HEV)',
            'Alcoholic hepatitis / cirrhosis',
            'Drug-induced liver injury (DILI)',
            'Autoimmune hepatitis',
            'Non-alcoholic steatohepatitis (NASH)',
            'Ischaemic hepatitis (shock liver)',
        ],
        'common_causes_cn': [
            '病毒性肝炎（乙、丙、甲、戊型）',
            '酒精性肝炎 / 肝硬化',
            '药物性肝损伤',
            '自身免疫性肝炎',
            '非酒精性脂肪性肝炎',
            '缺血性肝炎（休克肝）',
        ],
        'recommended_workup_en': [
            'Viral serology: HBsAg, anti-HCV, anti-HAV-IgM, anti-HEV-IgM',
            'Autoimmune markers: ANA, ASMA, anti-LKM, IgG',
            'Drug/alcohol history review',
            'Iron / copper studies if age <40 (ferritin, ceruloplasmin)',
            'Liver ultrasound / FibroScan',
            'Liver biopsy if diagnosis unclear',
        ],
        'recommended_workup_cn': [
            '病毒血清学：HBsAg、抗HCV、抗HAV-IgM、抗HEV-IgM',
            '自身免疫标志物：ANA、ASMA、抗LKM、IgG',
            '药物/酒精病史回顾',
            '<40岁查铁/铜代谢（铁蛋白、铜蓝蛋白）',
            '肝脏超声 / FibroScan',
            '诊断不明时肝穿刺活检',
        ],
        'department_en': 'Infectious Disease / Hepatology',
        'department_cn': '感染科 / 肝病科',
        'urgency': 'moderate',
    },
    'cholestatic': {
        'en_name': 'Cholestatic Jaundice',
        'cn_name': '胆汁郁积性黄疸',
        'mechanism': 'Bile flow obstruction, intra- or extra-hepatic.',
        'bilirubin_pattern': 'Direct bilirubin dominant (>50% of total).',
        'lab_pattern': 'ALP/GGT >> ALT/AST (ratio >3).',
        'common_causes_en': [
            'Choledocholithiasis (bile duct stones)',
            'Pancreatic head cancer',
            'Cholangiocarcinoma',
            'Primary biliary cholangitis (PBC)',
            'Primary sclerosing cholangitis (PSC)',
            'Biliary strictures (post-surgical, inflammatory)',
            'Drug-induced cholestasis',
        ],
        'common_causes_cn': [
            '胆总管结石',
            '胰头癌',
            '胆管癌',
            '原发性胆汁性胆管炎（PBC）',
            '原发性硬化性胆管炎（PSC）',
            '胆管狭窄（术后、炎症性）',
            '药物性胆汁淤积',
        ],
        'recommended_workup_en': [
            'Abdominal ultrasound (first-line: ducts, gallbladder, pancreas)',
            'MRCP (magnetic resonance cholangiopancreatography)',
            'Contrast CT abdomen if mass suspected',
            'Tumour markers: CA19-9, CEA',
            'AMA for PBC; p-ANCA for PSC',
            'ERCP (diagnostic and therapeutic)',
            'EUS for pancreatic lesions',
        ],
        'recommended_workup_cn': [
            '腹部超声（首选：评估胆管、胆囊、胰腺）',
            'MRCP（磁共振胰胆管造影）',
            '腹部增强CT（怀疑占位时）',
            '肿瘤标志物：CA19-9、CEA',
            'AMA筛查PBC；p-ANCA筛查PSC',
            'ERCP（诊断和治疗性）',
            'EUS（内镜超声）评估胰腺病变',
        ],
        'department_en': 'Gastroenterology / Hepatobiliary Surgery',
        'department_cn': '消化内科 / 肝胆外科',
        'urgency': 'high',
    },
    'hemolytic': {
        'en_name': 'Haemolytic Jaundice',
        'cn_name': '溶血性黄疸',
        'mechanism': 'Red cell destruction exceeds liver conjugation capacity.',
        'bilirubin_pattern': 'Indirect bilirubin dominant (>80% of total).',
        'lab_pattern': 'Indirect bilirubin elevation, normal ALT/AST/ALP, low haptoglobin.',
        'common_causes_en': [
            'Autoimmune haemolytic anaemia (AIHA)',
            'G6PD deficiency',
            'Hereditary spherocytosis / elliptocytosis',
            'Sickle cell disease / thalassaemia',
            'Mismatched blood transfusion',
            'Mechanical haemolysis (prosthetic valve, MAHA)',
            'Infections (malaria, Clostridium)',
        ],
        'common_causes_cn': [
            '自身免疫性溶血性贫血（AIHA）',
            'G6PD缺乏症',
            '遗传性球形/椭圆形红细胞增多症',
            '镰状细胞病 / 地中海贫血',
            '血型不合输血',
            '机械性溶血（人工瓣膜、微血管病性）',
            '感染（疟疾、梭状芽胞杆菌）',
        ],
        'recommended_workup_en': [
            'CBC with peripheral blood smear',
            'Reticulocyte count',
            'Haptoglobin (low in haemolysis)',
            'LDH (elevated)',
            'Direct Coombs test (DAT)',
            'G6PD enzyme assay',
            'Haemoglobin electrophoresis',
        ],
        'recommended_workup_cn': [
            '血常规 + 外周血涂片',
            '网织红细胞计数',
            '结合珠蛋白（溶血时降低）',
            'LDH（升高）',
            'Coombs试验（直接抗人球蛋白试验）',
            'G6PD酶活性测定',
            '血红蛋白电泳',
        ],
        'department_en': 'Hematology',
        'department_cn': '血液科',
        'urgency': 'moderate',
    },
}


# ═════════════════════════════════════════════════════════════
# 4. SEVERITY GUIDANCE  (clinical severity 0-3)
# ═════════════════════════════════════════════════════════════
# TBIL thresholds (μmol/L). Clinical severity also factors in INR / encephalopathy.

SEVERITY_GUIDANCE = {
    0: {  # Normal / borderline
        'en': {
            'label': 'Normal / Borderline',
            'tbil': '< 34',
            'management': 'No specific treatment needed; routine monitoring.',
            'follow_up': 'Repeat liver function in 2-4 weeks if borderline (TBIL 20-34).',
            'lifestyle': 'Avoid alcohol and hepatotoxic drugs; healthy diet.',
        },
        'cn': {
            'label': '正常 / 临界',
            'tbil': '< 34',
            'management': '无需特殊治疗，常规监测。',
            'follow_up': '临界值（TBIL 20-34）2-4周后复查肝功能。',
            'lifestyle': '避免饮酒和肝毒性药物，保持健康饮食。',
        },
    },
    1: {  # Mild
        'en': {
            'label': 'Mild Jaundice (TBIL 34-85)',
            'tbil': '34 - 85',
            'management': 'Outpatient evaluation recommended; identify the cause.',
            'follow_up': 'Liver function weekly until stable; imaging if persistent.',
            'lifestyle': 'Strict abstinence from alcohol; review all medications for hepatotoxicity.',
        },
        'cn': {
            'label': '轻度黄疸 (TBIL 34-85)',
            'tbil': '34 - 85',
            'management': '建议门诊评估，明确病因。',
            'follow_up': '每周复查肝功能至稳定，持续异常需影像学检查。',
            'lifestyle': '严格戒酒，审查所有药物的肝毒性。',
        },
    },
    2: {  # Moderate
        'en': {
            'label': 'Moderate Jaundice (TBIL 85-171)',
            'tbil': '85 - 171',
            'management': 'Inpatient admission advised; active diagnostic work-up.',
            'follow_up': 'Daily liver function; coagulation (PT/INR); watch decompensation.',
            'lifestyle': 'Hospital monitoring; IV fluids if dehydrated; nutritional support.',
        },
        'cn': {
            'label': '中度黄疸 (TBIL 85-171)',
            'tbil': '85 - 171',
            'management': '建议住院治疗，积极完善病因诊断。',
            'follow_up': '每日监测肝功能，检查凝血功能（PT/INR），警惕失代偿。',
            'lifestyle': '住院监护，必要时补液，给予营养支持。',
        },
    },
    3: {  # Severe
        'en': {
            'label': 'Severe Jaundice (TBIL > 171)',
            'tbil': '> 171',
            'management': 'Urgent admission; consider ICU; evaluate for acute/acute-on-chronic '
                          'liver failure and synthetic dysfunction.',
            'follow_up': 'Liver function q6-12h; INR, lactate, ammonia; screen for hepatic '
                          'encephalopathy; urine output.',
            'lifestyle': 'ICU-level care; do NOT restrict protein in encephalopathy '
                     '(small frequent meals + late-evening snack, branch-chain '
                     'amino acids if needed); lactulose.',
        },
        'cn': {
            'label': '重度黄疸 (TBIL > 171)',
            'tbil': '> 171',
            'management': '紧急住院，考虑ICU监护，评估急性/慢加急性肝衰竭及合成功能衰竭。',
            'follow_up': '每6-12小时监测肝功能，查INR、乳酸、血氨，筛查肝性脑病，记录尿量。',
            'lifestyle': 'ICU级别护理；肝性脑病时不限制蛋白（少量多餐+夜间加餐，必要时支链氨基酸），用乳果糖。',
        },
    },
}


# ═════════════════════════════════════════════════════════════
# 5. CLINICAL PATHWAYS  (acute red-flag bundles)
# ═════════════════════════════════════════════════════════════

CLINICAL_PATHWAYS = {
    'acute_liver_failure': {
        'en_trigger': 'INR > 1.5 + encephalopathy (+ TBIL high), without known chronic '
                       'liver disease; or coagulopathy with any grade of encephalopathy.',
        'cn_trigger': 'INR > 1.5 + 肝性脑病（+ TBIL升高），无明确慢性肝病；'
                       '或凝血障碍伴任何分级脑病。',
        'en_actions': [
            'ICU admission immediately',
            'N-acetylcysteine (NAC) regardless of cause',
            'Contact liver transplant centre',
            'Monitor INR q6h, ammonia, lactate, glucose',
            'Consider intracranial pressure monitoring',
            "Apply King's College Criteria for transplant",
        ],
        'cn_actions': [
            '立即入ICU',
            '无论病因给予N-乙酰半胱氨酸（NAC）',
            '联系肝移植中心',
            '监测：INR每6小时、血氨、乳酸、血糖',
            '考虑颅内压监测',
            'King学院标准评估肝移植指征',
        ],
        'guidelines': ['LF_2018', 'KINGS_COLLEGE_ALF', 'AASLD_ALF_2023', 'EASL_COAG_2022', 'HE_ENCEPHALOPATHY_2014'],
        'icon': '🚨',
    },
    'aclf': {
        'en_trigger': 'Acute decompensation of chronic liver disease with jaundice + '
                       'coagulopathy and/or organ failure (COSSH-ACLF / CLIF-C ACLF criteria).',
        'cn_trigger': '慢性肝病急性失代偿，伴黄疸+凝血障碍和/或器官衰竭（COSSH-ACLF / CLIF-C ACLF）。',
        'en_actions': [
            'Identify and remove precipitant (infection, GI bleeding, alcohol, hepatotoxic drugs, HBV reactivation)',
            'Start nucleos(t)ide analogues immediately for HBV-ACLF',
            'Score severity (COSSH-ACLF / CLIF-C ACLF / CLIF-C OF)',
            'Organ support; treat infection; albumin for SBP',
            'Early transplant evaluation for high-grade ACLF',
        ],
        'cn_actions': [
            '识别并去除诱因（感染、消化道出血、酒精、肝毒性药物、HBV再激活）',
            'HBV-ACLF立即启动核苷（酸）类似物',
            '评分（COSSH-ACLF / CLIF-C ACLF / CLIF-C OF）',
            '器官支持；抗感染；自发性细菌性腹膜炎用人血白蛋白',
            '高分级ACLF尽早评估肝移植',
        ],
        'guidelines': ['LF_2018', 'APASL_ACLF_2019', 'EASL_CLIF_2013', 'EASL_DECOMP_2018', 'SURVIVING_SEPSIS_2021', 'HE_ENCEPHALOPATHY_2014'],
        'icon': '🚨',
    },
    'acute_cholangitis': {
        'en_trigger': 'Fever + jaundice + abdominal pain (Charcot triad); add shock / '
                       'altered mentation = Reynolds pentad (severe/acute cholangitis).',
        'cn_trigger': '发热 + 黄疸 + 腹痛（Charcot三联征）；加休克/意识改变=Reynolds五联征'
                       '（重度/急性梗阻性化脓性胆管炎）。',
        'en_actions': [
            'Grade severity by Tokyo Guidelines (TG18)',
            'Blood cultures + broad-spectrum antibiotics early',
            'Biliary drainage: ERCP (first-line), PTCD, or surgery',
            'Resuscitate: fluids, vasopressors if septic shock',
            'Timing of drainage by grade (mild: urgent; moderate: early; severe: emergency)',
        ],
        'cn_actions': [
            '按东京指南(TG18)分级严重程度',
            '尽早血培养+广谱抗生素',
            '胆道引流：ERCP（首选）、PTCD或手术',
            '复苏：补液，脓毒症休克时用血管活性药',
            '按分级决定引流时机（轻：急诊；中：早期；重：紧急）',
        ],
        'guidelines': ['ACUTE_BILIARY_2021', 'TOKYO_TG18', 'SURVIVING_SEPSIS_2021', 'ACR_JAUNDICE'],
        'icon': '🚨',
    },
    'obstructive_jaundice': {
        'en_trigger': 'Direct bilirubin dominant + ALP > 3x ULN ± dilated ducts on imaging.',
        'cn_trigger': '直接胆红素升高为主 + ALP > 3倍正常上限 ± 影像见胆管扩张。',
        'en_actions': [
            'Confirm biliary obstruction by imaging first (ultrasound, then MRCP/CT)',
            'Urgent surgical / GI consult',
            'ERCP for biliary decompression if stones',
            'CT/MRI for malignancy staging if mass',
            'Pre-op: correct coagulopathy (vitamin K, FFP)',
            'Antibiotics if cholangitis suspected',
        ],
        'cn_actions': [
            '先影像确认胆道梗阻（超声首选，继而MRCP/CT）',
            '紧急外科/消化科会诊',
            'ERCP胆管减压（结石）',
            'CT/MRI肿瘤分期（占位）',
            '术前纠正凝血障碍（维生素K、FFP）',
            '怀疑胆管炎时使用抗生素',
        ],
        'guidelines': ['ACUTE_BILIARY_2021', 'TOKYO_TG18', 'ACR_JAUNDICE', 'EASL_COAG_2022'],
        'icon': '⚠️',
    },
    'hemolytic_crisis': {
        'en_trigger': 'Rapidly rising indirect bilirubin + anaemia + dark urine ± organ '
                       'dysfunction (haemolytic crisis).',
        'cn_trigger': '间接胆红素快速升高 + 贫血 + 尿色加深 ± 器官功能障碍（溶血危象）。',
        'en_actions': [
            'Stop offending drug / oxidative trigger',
            'Blood count, reticulocytes, haptoglobin, LDH, DAT',
            'Hydration ± transfusion support (cautiously in AIHA)',
            'Steroids/rituximab for AIHA; treat underlying cause',
            'Renal monitoring; ICU if organ failure',
        ],
        'cn_actions': [
            '停用可疑药物/氧化诱发因素',
            '血常规、网织红细胞、结合珠蛋白、LDH、Coombs试验',
            '补液±输血支持（AIHA谨慎输血）',
            'AIHA用糖皮质激素/利妥昔单抗；针对病因治疗',
            '监测肾功能；器官衰竭入ICU',
        ],
        'guidelines': ['TMA_HUS_GUIDANCE'],
        'icon': '⚠️',
    },
    'dili_urgent': {
        'en_trigger': 'DILI with jaundice + ALT > 3x ULN (Hy\'s law case) — high risk of '
                       'acute liver failure / fatal outcome.',
        'cn_trigger': '药物性肝损伤伴黄疸 + ALT > 3倍正常上限（Hy定律病例）'
                       '——急性肝衰竭/致死风险高。',
        'en_actions': [
            'Immediately discontinue the suspect drug and all non-essential drugs',
            'Causality assessment by RUCAM',
            'N-acetylcysteine if paracetamol-related or severe',
            'Monitor INR, encephalopathy; hepatology consult',
            'Report to pharmacovigilance',
        ],
        'cn_actions': [
            '立即停用可疑药物及所有非必需药物',
            'RUCAM因果评估',
            '对乙酰氨基酚相关或重症者用NAC',
            '监测INR、肝性脑病；肝病科会诊',
            '上报药物不良反应监测',
        ],
        'guidelines': ['DILI_2023', 'EASL_DILI_2019'],
        'icon': '⚠️',
    },
}


# ═════════════════════════════════════════════════════════════
# 6. CLINICAL ADVISOR
# ═════════════════════════════════════════════════════════════

# Map BilinGuard DBIL/IBIL grade_index buckets (mirror code/config thresholds)
DBIL_BUCKETS = {'Normal': 0, 'Mild Elevation': 1, 'Severe Elevation': 2}
IBIL_BUCKETS = {'Normal': 0, 'Mild Elevation': 1, 'Severe Elevation': 2}

DISCLAIMER = {
    'en': 'AI-assisted screening prompt only. BilinGuard does not make a diagnosis and '
          'must NOT replace clinical judgement, laboratory confirmation, or the treating '
          'physician\'s decision. Confirm all results with serum bilirubin and full work-up.',
    'cn': '本提示仅为AI辅助筛查参考。BilinGuard不作出诊断，不能替代临床判断、化验确认或'
          '主治医师决策。所有结果须以血清胆红素及完整检查为准。',
}

# Per-jaundice-type guideline evidence (used for the jaundice-type section)
TYPE_GUIDELINES = {
    'hepatocellular': ['CHB_2022', 'CIRRHOSIS_2019', 'DILI_2023', 'AIH_2021', 'ALD_2018',
                       'EASL_DILI_2019', 'EASL_AIH_2015', 'EASL_ARD_2024', 'EASL_NIT_2021'],
    'cholestatic': ['ACUTE_BILIARY_2021', 'TOKYO_TG18', 'PBC_2021', 'PSC_2023',
                    'WORLD_GALLSTONE_2022', 'EASL_PSC_2022', 'ACR_JAUNDICE', 'EASL_PBC_2017'],
    'hemolytic': ['TMA_HUS_GUIDANCE'],
}

# Child-Pugh grade interpretation (grade_index 0=A, 1=B, 2=C)
CP_INTERP = {
    0: {'en': 'Child-Pugh A (score 5-6): compensated, preserved hepatic reserve; '
              'tolerates liver resection / TACE.',
        'cn': 'Child-Pugh A（5-6 分）：代偿期，肝储备良好；可耐受肝切除 / TACE。'},
    1: {'en': 'Child-Pugh B (score 7-9): decompensated; surgical and TACE risk is elevated — '
              'cautious TACE only within B7, optimise reserve first.',
        'cn': 'Child-Pugh B（7-9 分）：失代偿；手术与 TACE 风险升高——仅 B7 内可谨慎 TACE，先优化肝储备。'},
    2: {'en': 'Child-Pugh C (score 10-15): severe decompensation; major surgery / TACE '
              'contraindicated; evaluate for liver transplantation.',
        'cn': 'Child-Pugh C（10-15 分）：严重失代偿；禁忌大手术 / TACE；评估肝移植。'},
}
# MELD risk interpretation (grade_index 0=Low, 1=Medium, 2=High)
MELD_INTERP = {
    0: {'en': 'MELD Low (<=20): 3-month mortality <6%; non-urgent, routine follow-up '
              'and aetiology treatment.',
        'cn': 'MELD Low（≤20）：3 个月死亡 <6%；非紧急，常规随访与病因治疗。'},
    1: {'en': 'MELD Medium (21-30): 3-month mortality ~20%; priority transplant evaluation; '
              'manage decompensation.',
        'cn': 'MELD Medium（21-30）：3 个月死亡 ~20%；优先肝移植评估；处理失代偿。'},
    2: {'en': 'MELD High (>30): 3-month mortality >50%; urgent liver-transplant evaluation; '
              'ICU-level monitoring.',
        'cn': 'MELD High（>30）：3 个月死亡 >50%；紧急肝移植评估；ICU 级别监护。'},
}
CP_MELD_GUIDELINES = ['CHILD_PUGH', 'MELD_3_0', 'MELD_3_0_2021', 'MELD_ORIGINAL_2001',
                      'MELD_NA_2006', 'ALBI_2015', 'CHINA_LIVER_TRANSPLANT_2018']

# ═════════════════════════════════════════════════════════════
# NUTRITION SUPPORT GUIDANCE (v3.1) — ESPEN liver-disease nutrition,
# driven by NRS-2002 screening + jaundice/severity/disease context.
# Anchored to ESPEN_LIVER_2019 and NRS2002_2003.
# ═════════════════════════════════════════════════════════════
NUTRITION_GUIDELINE_IDS = ['ESPEN_LIVER_2019', 'NRS2002_2003']

NUTRITION_RISK_THRESHOLD = 3   # NRS-2002 >= 3 → nutrition risk
NUTRITION_SCORE_MAX = 7        # valid NRS-2002 range 0-7

NUTRITION_GUIDANCE = {
    'at_risk': {
        'en': {
            'label': 'Nutrition risk (NRS-2002 >= 3) — intervene',
            'actions': [
                'Dietitian consult within 24-48 h; start nutrition intervention',
                'Targets: 30-35 kcal/kg/day (35-40 if malnourished); '
                'protein 1.2-1.5 g/kg/day',
                'Do NOT restrict protein (incl. hepatic encephalopathy — '
                'protein restriction is obsolete)',
                'Small frequent meals every 3-4 h + late-evening snack '
                '(~400-500 kcal, e.g. complex carbohydrate)',
                'Re-assess weekly; monitor weight, handgrip, intake diary',
            ],
        },
        'cn': {
            'label': '营养风险（NRS-2002 ≥ 3 分）——需干预',
            'actions': [
                '24-48 小时内营养科会诊，启动营养干预',
                '目标：30-35 kcal/kg/d（营养不良者 35-40）；蛋白 1.2-1.5 g/kg/d',
                '不限制蛋白摄入（含肝性脑病——蛋白限制已被淘汰）',
                '少量多餐（每 3-4 小时）+ 夜间加餐（约 400-500 kcal，如复合碳水）',
                '每周复评：体重、握力、进食记录',
            ],
        },
    },
    'no_risk': {
        'en': {
            'label': 'No current nutrition risk (NRS-2002 < 3)',
            'actions': [
                'Re-screen weekly during hospitalisation (NRS-2002)',
                'Ensure adequate oral intake; no prophylactic restriction',
            ],
        },
        'cn': {
            'label': '暂无营养风险（NRS-2002 < 3 分）',
            'actions': [
                '住院期间每周复评 NRS-2002',
                '保证经口摄入充足，不作预防性限制',
            ],
        },
    },
    'unscreened': {
        'en': {
            'label': 'Nutrition screening not performed',
            'actions': [
                'Complete NRS-2002 screening (incl. BMI, recent weight loss, '
                'intake change, disease severity)',
                'Sarcopenia vigilance in chronic liver disease even when BMI normal',
            ],
        },
        'cn': {
            'label': '尚未完成营养风险筛查',
            'actions': [
                '完成 NRS-2002 筛查（含 BMI、近期体重下降、进食变化、疾病严重程度）',
                '慢性肝病即使 BMI 正常也需警惕肌少症',
            ],
        },
    },
}

# Severity addenda (protein/energy targets per ESPEN liver guideline)
NUTRITION_SEVERITY_ADDENDA = {
    3: {
        'en': 'Severe/ACLF context: favour early enteral nutrition '
              '(35-40 kcal/kg/day); monitor refeeding syndrome.',
        'cn': '重度/ACLF：优先早期肠内营养（35-40 kcal/kg/d）；警惕再喂养综合征。',
    },
    2: {
        'en': 'Moderate disease: ensure 30-35 kcal/kg/day and 1.2 g/kg/day '
              'protein minimum; late-evening snack advised in cirrhosis.',
        'cn': '中度：保证 30-35 kcal/kg/d、蛋白至少 1.2 g/kg/d；肝硬化建议夜间加餐。',
    },
}

# Jaundice-type addenda
NUTRITION_TYPE_ADDENDA = {
    'cholestatic': {
        'en': 'Cholestasis/obstruction: fat malabsorption — use medium-chain '
              'triglycerides (MCT); supplement fat-soluble vitamins A/D/E/K '
              '(K especially when INR prolonged).',
        'cn': '胆汁郁积/梗阻：脂肪吸收不良——改用中链甘油三酯（MCT）；补充脂溶性维生素 '
              'A/D/E/K（INR 延长时尤其补 K）。',
    },
    'hepatocellular': {
        'en': 'Hepatocellular disease: late-evening snack + no protein '
              'restriction; branch-chain amino acid supplement if intolerant '
              'or encephalopathic.',
        'cn': '肝细胞型肝病：夜间加餐 + 不限蛋白；不耐受或肝性脑病时可补充支链氨基酸。',
    },
    'hemolytic': {
        'en': 'Haemolysis: no specific nutrition modification; support '
              'erythropoiesis (adequate folate/iron via normal diet).',
        'cn': '溶血：无特殊营养调整；普通饮食保证叶酸/铁充足以支持造血。',
    },
}


# ═════════════════════════════════════════════════════════════
# AGENT PIPELINE (v3): perceive → route → reason → critique → explain
# Design anchors (frontier 2024-2026 literature):
#   - case-grounded structured perception + decision-mode routing
#     (HemaGuide, Nature Medicine 2026, PMID 42380678)
#   - machine-readable rule triggers with GRADE-style strength
#     (OnCATs, Int J Med Inform 2026, PMID 41124940)
#   - pre-output self-critique, transparency & monitoring fields
#     (JAMIA 2024 responsible AI-CDSS recommendations, PMID 39325508)
#   - auditable execution trace / evidence-grounded reports
#     (AgentEYE, Cell Reports Medicine 2026, PMID 42556343)
#   - alert prioritisation against alert fatigue
#     (Ochsner J 2014, PMID 24940129; Swiss Med Wkly 2023, PMID 37454289)
# ═════════════════════════════════════════════════════════════
AGENT_PIPELINE_VERSION = 'v3-agent'

# Labs whose absence blocks safety-critical rule evaluation
CRITICAL_LABS = ('tbil', 'dbil', 'inr', 'albumin')

# GRADE-style strength per rule domain: (grade, strength, evidence quality)
DOMAIN_GRADE = {
    'red_flag':          ('1B', 'strong',     'moderate'),
    'severity':          ('2B', 'conditional', 'moderate'),
    'type_classification': ('2A', 'conditional', 'high'),
    'disease_routing':   ('2C', 'conditional', 'low'),
    'routing':           ('2C', 'conditional', 'low'),
    'lab_interpretation': ('2A', 'conditional', 'high'),
    'risk_stratification': ('1A', 'strong',     'high'),
    'triage':            ('2B', 'conditional', 'moderate'),
    'nutrition':         ('1B', 'strong',     'moderate'),
}

# Classical validated scoring instruments — age is not a currency defect
_CLASSICAL_GIDS = {
    'CHILD_PUGH', 'KINGS_COLLEGE_ALF', 'MELD_ORIGINAL_2001', 'MELD_NA_2006',
    'EASL_CLIF_2013', 'ALBI_2015', 'HE_ENCEPHALOPATHY_2014', 'NRS2002_2003',
}


def _compute_stale_guidelines(year_now):
    """Guideline ids older than 5 years, excluding classical instruments."""
    stale = set()
    for gid, entry in GUIDELINES.items():
        try:
            year = int(entry[4])
        except (TypeError, ValueError, IndexError):
            continue
        if year <= year_now - 5 and gid not in _CLASSICAL_GIDS:
            stale.add(gid)
    return stale


import datetime as _datetime
STALE_GUIDELINES = _compute_stale_guidelines(_datetime.date.today().year)


def _num(v, default=None):
    """Coerce a lab/form value to float; non-numeric, NaN and inf -> default.

    Robustness guard for real-world form input (strings, lists, None, NaN).
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float('inf'), float('-inf')):
        return default
    return f


class ClinicalAdvisor:
    """Combines BilinGuard AI predictions with a guideline-anchored knowledge base.

    Parameters
    ----------
    lang : str
        'en' or 'cn'.
    department : str, optional
        One of the ``DEPARTMENTS`` keys.  When set, advice is tailored to the
        clinician's specialty (referrals, monitoring, risks).  May be overridden
        per-call via ``advise(department=...)``.
    """

    def __init__(self, lang='en', department=None, audit_log=None):
        self.lang = lang
        self.department = department
        # Optional closed-loop audit trail (JSONL). Records are PHI-safe:
        # no free-text diagnosis, no patient identifiers, no lab values —
        # only decision metadata (mode, severity, rules, priority, confidence).
        self.audit_log = audit_log

    # ---- helpers ---------------------------------------------------------
    def _t(self, en, cn):
        return cn if self.lang == 'cn' else en

    def _list(self, d, base_key):
        """Pick the language list from a dict, key f'{base_key}_{lang}'."""
        return d.get(f'{base_key}_{self.lang}', d.get(f'{base_key}_en', []))

    # ---- main entry ------------------------------------------------------
    def advise(self, bilinguard_results, lab_values=None, department=None,
               diagnosis=None, nutrition=None):
        """Generate clinical guidance from a BilinGuard result.

        Parameters
        ----------
        bilinguard_results : dict
            Keys 'screen', 'grade', 'dbil', 'ibil', 'type' (as produced by the
            inference pipeline).  Any subset is allowed.
        lab_values : dict, optional
            alt, ast, alp, ggt, tbil, dbil, ibil, inr, albumin, ...
        department : str, optional
            Override of the constructor ``department``.
        diagnosis : str, optional
            Free-text admission diagnosis.  Used to infer the likely disease(s)
            and tailor advice.
        nutrition : dict or int, optional
            Nutrition screening input.  Either an int NRS-2002 total score
            (valid 0-7) or a dict with key ``nrs2002`` (plus optional
            ``bmi``, ``weight_loss_6m`` — currently informational).  Drives
            ESPEN-anchored nutrition support advice (v3.1).

        Returns
        -------
        dict with:
          - ``sections`` : list of {title, icon, content} (backward compatible)
          - ``summary``  : one-line triage recommendation
          - ``severity`` : 0-3
          - ``jaundice_type`` : hepatocellular / cholestatic / hemolytic / None
          - ``disease`` : primary disease id (or None)
          - ``diseases_matched`` : list of all matched disease ids
          - ``department`` : effective department
          - ``guidelines`` : list of {gid, short, citation}
          - ``disclaimer`` : str
          - ``agent`` : v3 agent payload (backward-compatible addition) with
            ``pipeline_version``, ``mode`` (normal / routine / red_flag /
            conflict / insufficient_data), ``case`` (perceived structured
            state), ``triggered_rules`` (rule_id, domain, trigger, action,
            urgency, GRADE strength, guideline ids), ``self_check``
            (consistency checks, alert priority, confidence, limitations) and
            ``trace`` (stage-by-stage execution log).
        """
        results = bilinguard_results or {}
        labs = lab_values or {}
        dept = department or self.department
        if dept and dept not in DEPARTMENTS:
            dept = None

        # 0. Nutrition screening state (NRS-2002, v3.1)
        nutrition_state, nrs_score = self._nutrition_state(nutrition)

        # 1. Severity (labs take precedence when available)
        severity = self._assess_severity(results, labs)
        jtype = self._determine_type(results, labs)

        # 2. Disease inference from admission diagnosis text
        disease_id, all_matched = self._infer_disease(diagnosis) if diagnosis else (None, [])

        # 3. Red-flag pathways
        pathway = self._check_pathways(severity, labs, disease_id, jtype)

        # 4. Effective department: explicit > disease-default > None
        eff_dept = dept
        if not eff_dept and disease_id:
            eff_dept = DISEASES[disease_id].get('department')
        if not eff_dept and jtype:
            eff_dept = {
                'hepatocellular': 'infectious_hepatology',
                'cholestatic': 'gi',
                'hemolytic': 'hematology',
            }.get(jtype)

        # ---- assemble sections --------------------------------------------
        sections = []

        # 4a. Severity
        sev_info = SEVERITY_GUIDANCE.get(severity, SEVERITY_GUIDANCE[1])
        sections.append({
            'title': self._t('Severity Assessment', '严重程度评估'),
            'content': sev_info[self.lang],
            'icon': '🚨' if severity >= 3 else ('⚠️' if severity >= 2 else '✅'),
            'guidelines': self._severity_guidelines(severity),
        })

        # 4b. Jaundice type
        if jtype:
            type_info = JAUNDICE_TYPES[jtype]
            type_content = {
                'mechanism': type_info['mechanism'],
                'pattern': type_info['bilirubin_pattern'],
                'lab_pattern': type_info['lab_pattern'],
                'causes': self._list(type_info, 'common_causes'),
                'workup': self._list(type_info, 'recommended_workup'),
                'department': type_info.get(f'department_{self.lang}', ''),
            }
            v_note = self._type_verification_note(results, labs, jtype)
            if v_note:
                type_content['verification'] = v_note
            sections.append({
                'title': f'{self._t("Jaundice Type", "黄疸类型")}: '
                         f'{type_info[f"{self.lang}_name"]}',
                'content': type_content,
                'icon': '🏥',
                'guidelines': self._gids_to_objs(TYPE_GUIDELINES.get(jtype, [])),
            })

        # 4c. Disease (inferred)
        if disease_id:
            dis = DISEASES[disease_id]
            sections.append({
                'title': f'{self._t("Likely Aetiology", "疑似病因")}: '
                         f'{dis[f"name_{self.lang}"]}',
                'content': {
                    'principle': dis.get(f'principle_{self.lang}',
                                         dis.get('principle_en', '')),
                    'jaundice_type': JAUNDICE_TYPES.get(dis.get('jaundice_type'), {}).get(
                        f'{self.lang}_name', dis.get('jaundice_type', '')),
                    'suggested_department': DEPARTMENTS.get(
                        dis.get('department'), {}).get(f'name_{self.lang}', ''),
                },
                'icon': '🩺',
                'guidelines': self._gids_to_objs(dis.get('guidelines', [])),
            })

        # 4c-bis. Child-Pugh / MELD risk stratification (from BilinGuard CP/MELD models)
        cp_meld_sec = self._cp_meld_section(results)
        if cp_meld_sec:
            sections.append(cp_meld_sec)

        # 4d. Department-tailored guidance
        if eff_dept:
            sections.append(self._department_section(eff_dept, jtype, disease_id, severity))

        # 4e. Red-flag pathway alert (highest priority visually)
        if pathway:
            sections.append({
                'title': f'{self._t("Clinical Alert", "临床警示")}: '
                         f'{pathway["en_trigger" if self.lang == "en" else "cn_trigger"]}',
                'content': pathway[f'{self.lang}_actions'],
                'icon': pathway.get('icon', '🚨'),
                'guidelines': self._gids_to_objs(pathway.get('guidelines', [])),
            })

        # 4f. Laboratory interpretation
        if labs:
            lab_interp = self._interpret_labs(labs, jtype)
            if lab_interp:
                sections.append({
                    'title': self._t('Laboratory Interpretation', '化验解读'),
                    'content': lab_interp,
                    'icon': '🔬',
                    'guidelines': [],
                })

        # 4f-bis. Nutrition support guidance (v3.1, ESPEN-anchored)
        clinical_context = bool(jtype or severity >= 1 or labs
                                or (results.get('screen') or {}).get('is_jaundice')
                                or results.get('binary'))
        if nutrition_state == 'at_risk' or nutrition_state == 'no_risk' or clinical_context:
            nut_sec = self._nutrition_section(nutrition_state, nrs_score,
                                              severity, jtype,
                                              weight=self._nutrition_weight(nutrition))
            if nut_sec:
                sections.append(nut_sec)

        # 4g. Referral / triage recommendation
        sections.append(self._triage_section(jtype, severity, eff_dept,
                                             disease_id, pathway))

        # ---- top-level summary + metadata --------------------------------
        summary = self._build_summary(jtype, severity, eff_dept, disease_id, pathway)

        guidelines = self._collect_guidelines(disease_id, pathway, eff_dept, jtype)

        agent_payload = self._agent_run(results, labs, diagnosis, dept,
                                        severity, jtype, disease_id, all_matched,
                                        pathway, eff_dept,
                                        nutrition_state=nutrition_state,
                                        nrs_score=nrs_score)

        if self.audit_log:
            self._write_audit_record({
                'pipeline_version': AGENT_PIPELINE_VERSION,
                'mode': agent_payload['mode'],
                'severity': severity,
                'jaundice_type': jtype,
                'disease': disease_id,
                'department': eff_dept,
                'alert_priority': agent_payload['self_check']['alert_priority'],
                'confidence': agent_payload['self_check']['confidence'],
                'rule_ids': [r['rule_id'] for r in agent_payload['triggered_rules']],
                'n_rules': len(agent_payload['triggered_rules']),
                'nutrition_state': nutrition_state,
                'lang': self.lang,
            })

        return {
            'sections': sections,
            'summary': summary,
            'severity': severity,
            'jaundice_type': jtype,
            'disease': disease_id,
            'diseases_matched': all_matched,
            'department': eff_dept,
            'guidelines': guidelines,
            'disclaimer': DISCLAIMER[self.lang],
            'agent': agent_payload,
        }

    # ---- severity --------------------------------------------------------
    def _assess_severity(self, results, labs):
        """0 normal, 1 mild, 2 moderate, 3 severe. Labs take precedence; CP/MELD escalates."""
        tbil = _num(labs.get('tbil'))

        # Red-flag overrides → severe regardless of TBIL
        inr = _num(labs.get('inr'))
        if inr is not None and inr > 1.5:
            sev = 3
        elif tbil is not None:
            if tbil > 171:
                sev = 3
            elif tbil >= 85:
                sev = 2
            elif tbil >= 34:
                sev = 1
            else:
                sev = 0
        else:
            # Fallback to BilinGuard grading
            grade = results.get('grade', {})
            if 'grading' in grade:
                gi = grade['grading'].get('grade_index', 0)
                try:
                    gi = int(gi)
                except (TypeError, ValueError):
                    gi = 0
                sev = min(gi + 1, 3)  # mild=1, moderate=2, severe=3
            else:
                dbil = results.get('dbil', {})
                if 'grading' in dbil:
                    try:
                        sev = int(dbil['grading'].get('grade_index', 0))
                    except (TypeError, ValueError):
                        sev = 0
                else:
                    screen = results.get('screen', {})
                    sev = 1 if screen.get('is_jaundice', False) else 0

        # Child-Pugh / MELD escalation (model outputs)
        sev = max(sev, self._cp_meld_floor(results))
        return sev

    def _cp_meld_floor(self, results):
        """Minimum clinical severity implied by Child-Pugh / MELD model outputs."""
        floor = 0
        cp = results.get('cp', {})
        if isinstance(cp, dict) and 'grading' in cp:
            try:
                cgi = int(cp['grading'].get('grade_index', -1))
            except (TypeError, ValueError):
                cgi = -1
            if cgi == 2:
                floor = max(floor, 3)   # CP-C → severe
            elif cgi == 1:
                floor = max(floor, 2)   # CP-B → at least moderate
        meld = results.get('meld', {})
        if isinstance(meld, dict) and 'grading' in meld:
            try:
                mgi = int(meld['grading'].get('grade_index', -1))
            except (TypeError, ValueError):
                mgi = -1
            if mgi == 2:
                floor = max(floor, 3)   # MELD High → severe
            elif mgi == 1:
                floor = max(floor, 2)   # MELD Medium → at least moderate
        return floor

    def _cp_meld_section(self, results):
        """Build the Child-Pugh / MELD risk-stratification advice section from model outputs."""
        cp = results.get('cp') or {}
        meld = results.get('meld') or {}
        has_cp = isinstance(cp, dict) and 'grading' in cp
        has_meld = isinstance(meld, dict) and 'grading' in meld
        if not (has_cp or has_meld):
            return None
        content = {}
        if has_cp:
            gr = cp['grading']
            try:
                cgi = int(gr.get('grade_index', 1))
            except (TypeError, ValueError):
                cgi = 1
            content['child_pugh'] = (f"{gr.get('prediction', '?')} — "
                                     f"{CP_INTERP.get(cgi, CP_INTERP[1])[self.lang]}")
        if has_meld:
            gr = meld['grading']
            try:
                mgi = int(gr.get('grade_index', 1))
            except (TypeError, ValueError):
                mgi = 1
            content['meld'] = (f"{gr.get('prediction', '?')} — "
                               f"{MELD_INTERP.get(mgi, MELD_INTERP[1])[self.lang]}")
        # combined implication
        cgi = mgi = -1
        if has_cp:
            try: cgi = int(cp['grading'].get('grade_index', -1))
            except Exception: pass
        if has_meld:
            try: mgi = int(meld['grading'].get('grade_index', -1))
            except Exception: pass
        if cgi == 2 or mgi == 2:
            content['implication'] = self._t(
                'Severe reserve / high short-term mortality → avoid surgery/TACE, urgent transplant evaluation.',
                '肝储备差 / 短期死亡风险高 → 避免手术与 TACE，紧急肝移植评估。')
        elif cgi == 1 or mgi == 1:
            content['implication'] = self._t(
                'Moderate reserve / elevated risk → optimise liver function before any invasive step.',
                '肝储备中度 / 风险升高 → 任何有创操作前先优化肝功能。')
        else:
            content['implication'] = self._t(
                'Preserved reserve / low short-term mortality → standard aetiology-directed management.',
                '肝储备尚可 / 短期死亡风险低 → 标准病因导向处理。')
        return {
            'title': self._t('Child-Pugh / MELD Risk Stratification', 'Child-Pugh / MELD 风险分层'),
            'content': content,
            'icon': '⚖️',
            'guidelines': self._gids_to_objs(CP_MELD_GUIDELINES),
        }

    # ---- jaundice type ---------------------------------------------------
    def _determine_type(self, results, lab_values=None):
        # 1) BilinGuard type classification
        type_r = results.get('type', {})
        if isinstance(type_r, dict) and 'type_classification' in type_r:
            ti = type_r['type_classification'].get('type_index', 0)
            try:
                ti = int(ti)
            except (TypeError, ValueError):
                ti = 0
            return 'hepatocellular' if ti == 0 else 'cholestatic'
        # If type_r is an int (0=hepato, 1=chol), use it directly
        if isinstance(type_r, (int, float)):
            return 'hepatocellular' if int(type_r) == 0 else 'cholestatic'

        # 2) Lab pattern
        if lab_values:
            dbil = _num(lab_values.get('dbil'), 0.0) or 0.0
            ibil = _num(lab_values.get('ibil'), 0.0) or 0.0
            total = dbil + ibil
            if total > 0:
                if dbil / total > 0.5:
                    return 'cholestatic'
                if ibil / total > 0.8:
                    return 'hemolytic'
                return 'hepatocellular'

        # 3) DBIL/IBIL grade fallback
        dbil_r = results.get('dbil', {}).get('grading', {})
        ibil_r = results.get('ibil', {}).get('grading', {})
        try:
            di = int(dbil_r.get('grade_index', 0))
            ii = int(ibil_r.get('grade_index', 0))
        except (TypeError, ValueError):
            di, ii = 0, 0
        if di > ii:
            return 'cholestatic'
        if ii > 1:
            return 'hemolytic'
        return 'hepatocellular'

    def _type_verification_note(self, results, labs, jtype):
        """Reviewer-driven P1 fix: qualify AI-derived jaundice type when
        laboratory confirmation is absent or discordant (blind-review theme
        '类型判定需化验验证', 15/60 comments)."""
        type_r = (results or {}).get('type')
        ai_sourced = (isinstance(type_r, dict) and 'type_classification' in type_r) \
            or isinstance(type_r, (int, float))
        if not ai_sourced:
            return None
        dbil = _num((labs or {}).get('dbil'))
        ibil = _num((labs or {}).get('ibil'))
        if dbil is None or ibil is None or (dbil + ibil) <= 0:
            return self._t(
                'Type classification is AI-derived; confirm with direct/'
                'indirect bilirubin fractionation.',
                '类型判定基于 AI，建议结合直接/间接胆红素分型验证。')
        total = dbil + ibil
        lab_type = ('cholestatic' if dbil / total > 0.5
                    else 'hemolytic' if ibil / total > 0.8
                    else 'hepatocellular')
        if lab_type == jtype:
            return None
        lab_name = JAUNDICE_TYPES.get(lab_type, {}).get(f'{self.lang}_name', lab_type)
        return self._t(
            f'AI type and laboratory pattern disagree (labs suggest '
            f'{lab_type}); re-check fractionation before type-directed action.',
            f'AI 类型与化验模式不一致（化验提示{lab_name}），'
            f'建议复核胆红素分型后再执行类型导向处理。')

    # ---- disease inference (keyword match on admission diagnosis) --------
    def _infer_disease(self, diagnosis_text):
        if not diagnosis_text:
            return None, []
        text = str(diagnosis_text).lower()
        matched = []
        for did, d in DISEASES.items():
            kws = [k.lower() for k in d.get('keywords', [])]
            if not any(k and k in text for k in kws):
                continue
            # apply negative-context exclusions
            exs = [e.lower() for e in d.get('exclude', [])]
            if any(e and e in text for e in exs):
                continue
            matched.append(did)
        if not matched:
            return 'unexplained', ['unexplained']
        # Highest priority wins as primary
        matched.sort(key=lambda d: DISEASES[d].get('priority', 0), reverse=True)
        return matched[0], matched

    # ---- red-flag pathways ----------------------------------------------
    def _check_pathways(self, severity, labs, disease_id, jtype):
        inr = _num(labs.get('inr'))
        dbil = _num(labs.get('dbil'))
        alp = _num(labs.get('alp'))

        # Disease-linked pathway (highest specificity)
        if disease_id:
            pid = DISEASES[disease_id].get('pathway')
            if pid and pid in CLINICAL_PATHWAYS:
                return CLINICAL_PATHWAYS[pid]

        # Lab-driven pathways
        if inr is not None and inr > 1.5 and severity >= 2:
            return CLINICAL_PATHWAYS['acute_liver_failure']

        if alp is not None and dbil is not None and alp > 300 and dbil > 50:
            return CLINICAL_PATHWAYS['obstructive_jaundice']

        if severity >= 3 and not labs:
            return CLINICAL_PATHWAYS.get('acute_liver_failure')

        return None

    # ---- department section ---------------------------------------------
    def _department_section(self, dept, jtype, disease_id, severity):
        info = DEPARTMENTS[dept]
        content = {
            'specialty': info[f'name_{self.lang}'],
            'focus': info[f'focus_{self.lang}'],
            'on_jaundice': self._list(info, 'on_jaundice'),
            'referral': info[f'referral_{self.lang}'],
        }
        # Type-specific decision branch (hepatocellular/cholestatic/hemolytic)
        type_dec = DEPARTMENT_TYPE_DECISIONS.get(dept, {}).get(self.lang, {}).get(jtype)
        if type_dec:
            type_name = JAUNDICE_TYPES.get(jtype, {}).get(f'{self.lang}_name', jtype)
            content['type_decision'] = f'【{type_name}】{type_dec}'
        return {
            'title': f'{self._t("Department Guidance", "科室建议")}: {info[f"name_{self.lang}"]}',
            'content': content,
            'icon': '🧭',
            'guidelines': self._gids_to_objs(info.get('guidelines', [])),
        }

    # ---- triage section --------------------------------------------------
    def _triage_section(self, jtype, severity, dept, disease_id, pathway=None):
        # Determine destination + urgency
        if severity >= 3:
            setting = self._t('ICU / emergency inpatient', 'ICU / 紧急住院')
            urgency = self._t('EMERGENT', '紧急')
            icon = '🚨'
        elif severity == 2:
            setting = self._t('Inpatient admission', '住院治疗')
            urgency = self._t('URGENT', '优先')
            icon = '⚠️'
        elif severity == 1:
            setting = self._t('Outpatient / short-stay evaluation', '门诊 / 短时留观评估')
            urgency = self._t('ROUTINE', '常规')
            icon = '📋'
        else:
            setting = self._t('No specific intervention; routine monitoring', '无需特殊处理，常规监测')
            urgency = self._t('NONE', '无需')
            icon = '✅'

        # Reviewer-driven P1 fix (blind-review theme '处置保守/激进'): when a
        # red-flag pathway is active, low-lab-severity must NOT be reported as
        # "no intervention" — escalate to prompt evaluation.
        if pathway and severity < 2:
            setting = self._t(
                'Prompt evaluation within 24-48 h (clinical alert pathway '
                'active despite mild labs)',
                '24-48 小时内限期评估（警示路径活跃，化验尚轻）')
            urgency = self._t('PRIORITY', '优先')
            icon = '⚠️'

        dest = DEPARTMENTS.get(dept, {}).get(f'name_{self.lang}') if dept else \
            (JAUNDICE_TYPES.get(jtype, {}).get(f'department_{self.lang}', '')
             if jtype else self._t('Specialist clinic', '专科门诊'))
        disease_name = DISEASES.get(disease_id, {}).get(f'name_{self.lang}', '')

        lines = [
            f'{self._t("Setting", "处置场所")}: {setting}',
            f'{self._t("Urgency", "紧急程度")}: {urgency}',
            f'{self._t("Refer to", "转诊至")}: {dest}',
        ]
        if disease_name:
            lines.append(f'{self._t("Working diagnosis", "初步诊断")}: {disease_name}')
        return {
            'title': self._t('Triage & Referral', '分诊与转诊'),
            'content': '\n'.join(lines),
            'icon': icon,
            'guidelines': self._collect_guidelines(disease_id, None, dept, jtype),
        }

    # ---- one-line summary ------------------------------------------------
    def _build_summary(self, jtype, severity, dept, disease_id, pathway):
        sev_label = SEVERITY_GUIDANCE.get(severity, SEVERITY_GUIDANCE[1])[self.lang]['label']
        type_name = JAUNDICE_TYPES.get(jtype, {}).get(f'{self.lang}_name', '')
        dis_name = DISEASES.get(disease_id, {}).get(f'name_{self.lang}', '')
        if self.lang == 'cn':
            parts = [sev_label]
            if type_name:
                parts.append(type_name)
            if dis_name:
                parts.append(f'疑似“{dis_name}”')
            if pathway:
                parts.append('⚠ 触发临床警示')
            return '｜'.join(parts) + '。请结合临床综合判断。'
        else:
            parts = [sev_label]
            if type_name:
                parts.append(type_name)
            if dis_name:
                parts.append(f'likely "{dis_name}"')
            if pathway:
                parts.append('⚠ clinical alert triggered')
            return ' | '.join(parts) + '. Use clinical judgement.'

    # ---- guideline collection -------------------------------------------
    def _gids_to_objs(self, gids):
        """Convert a list of guideline ids to structured citation objects, de-dup."""
        seen = set()
        out = []
        for g in gids:
            if g in GUIDELINES and g not in seen:
                seen.add(g)
                out.append({
                    'gid': g,
                    'short': GUIDELINES[g][0],
                    'citation': GUIDELINES[g][2 if self.lang == 'cn' else 1],
                    'source': GUIDELINES[g][3],
                    'year': GUIDELINES[g][4],
                })
        return out

    def _collect_guidelines(self, disease_id, pathway, dept, jtype):
        gids = []
        if disease_id:
            gids += DISEASES[disease_id].get('guidelines', [])
        if pathway:
            gids += pathway.get('guidelines', [])
        if dept:
            gids += DEPARTMENTS[dept].get('guidelines', [])
        if jtype == 'cholestatic':
            gids += ['ACUTE_BILIARY_2021', 'TOKYO_TG18']
        elif jtype == 'hepatocellular':
            gids += ['CHB_2022', 'CIRRHOSIS_2019']
        return self._gids_to_objs(gids)

    # ---- per-section guideline evidence ---------------------------------
    def _severity_guidelines(self, severity):
        gids = ['CHILD_PUGH', 'MELD_3_0']
        if severity >= 3:
            gids += ['LF_2018', 'AASLD_ALF_2023']
        elif severity >= 2:
            gids += ['CIRRHOSIS_2019']
        return self._gids_to_objs(gids)


    # ---- lab interpretation ---------------------------------------------
    def _interpret_labs(self, labs, jtype):
        interpretations = []
        ranges = {
            'alt': (7, 40), 'ast': (13, 35), 'alp': (44, 147),
            'ggt': (11, 50), 'tbil': (3.4, 20.5), 'dbil': (0, 6.8),
            'ibil': (1.7, 10.2), 'inr': (0.8, 1.2), 'albumin': (35, 50),
        }
        units = {
            'alt': 'U/L', 'ast': 'U/L', 'alp': 'U/L', 'ggt': 'U/L',
            'tbil': 'μmol/L', 'dbil': 'μmol/L', 'ibil': 'μmol/L',
            'inr': '', 'albumin': 'g/L',
        }
        names = {
            'en': {'alt': 'ALT', 'ast': 'AST', 'alp': 'ALP', 'ggt': 'GGT',
                   'tbil': 'Total Bilirubin', 'dbil': 'Direct Bilirubin',
                   'ibil': 'Indirect Bilirubin', 'inr': 'INR', 'albumin': 'Albumin'},
            'cn': {'alt': 'ALT', 'ast': 'AST', 'alp': 'ALP', 'ggt': 'GGT',
                   'tbil': '总胆红素', 'dbil': '直接胆红素', 'ibil': '间接胆红素',
                   'inr': 'INR', 'albumin': '白蛋白'},
        }
        for key in ranges:
            val = _num(labs.get(key))
            if val is None:
                continue
            lo, hi = ranges[key]
            name = names[self.lang].get(key, key)
            unit = units[key]
            status = self._t('normal', '正常')
            if val > hi:
                ratio = val / hi
                if ratio > 10:
                    status = self._t('CRITICAL', '危急')
                elif ratio > 3:
                    status = self._t('Severely elevated', '重度升高')
                elif ratio > 2:
                    status = self._t('Moderately elevated', '中度升高')
                else:
                    status = self._t('Mildly elevated', '轻度升高')
            elif val < lo and key in ('albumin',):
                status = self._t('Low', '降低')
            interpretations.append(f'{name}: {val:.1f} {unit} ({status})'.strip())

        # Pattern interpretation
        alp = _num(labs.get('alp'))
        alt = _num(labs.get('alt'))
        if jtype == 'cholestatic' and alp and alt:
            if alp / max(alt, 1) > 3:
                interpretations.append(self._t(
                    '→ Cholestatic pattern confirmed (ALP/ALT > 3)',
                    '→ 胆汁郁积型确认（ALP/ALT > 3）'))
        elif jtype == 'hepatocellular' and alp and alt:
            if alt / max(alp, 1) > 2:
                interpretations.append(self._t(
                    '→ Hepatocellular pattern confirmed (ALT/ALP > 2)',
                    '→ 肝细胞型确认（ALT/ALP > 2）'))

        # D → T ratio
        dbil = _num(labs.get('dbil'))
        tbil = _num(labs.get('tbil'))
        if dbil is not None and tbil is not None and tbil > 0:
            ratio = dbil / tbil
            tag = (self._t('direct-dominant (cholestatic/obstructive)',
                           '直接胆红素为主（胆汁郁积/梗阻性）')
                   if ratio > 0.5 else
                   self._t('indirect-dominant (haemolytic/pre-hepatic)',
                           '间接胆红素为主（溶血性/肝前性）'))
            interpretations.append(f'{self._t("D/T ratio", "直/总比")}: '
                                   f'{ratio:.0%} → {tag}')
        return interpretations


    # ---- nutrition support (v3.1) ----------------------------------------
    def _nutrition_state(self, nutrition):
        """Parse nutrition input -> (state, score).

        state: 'at_risk' (NRS-2002 >= 3) / 'no_risk' (valid < 3) /
        'unscreened' (absent or invalid).  Valid NRS-2002 range is 0-7.
        Accepts optional 'weight' (kg, 30-250) for individualised targets.
        """
        if nutrition is None:
            return 'unscreened', None
        if isinstance(nutrition, (int, float)):
            score = nutrition
        elif isinstance(nutrition, dict):
            try:
                score = float(nutrition.get('nrs2002'))
            except (TypeError, ValueError):
                return 'unscreened', None
        else:
            return 'unscreened', None
        if score != score or score < 0 or score > NUTRITION_SCORE_MAX:
            return 'unscreened', None
        score = int(score)
        return ('at_risk' if score >= NUTRITION_RISK_THRESHOLD else 'no_risk'), score

    @staticmethod
    def _nutrition_weight(nutrition):
        """Optional patient weight (kg) for individualised ESPEN targets."""
        if isinstance(nutrition, dict):
            w = _num(nutrition.get('weight'))
            if w is not None and 30 <= w <= 250:
                return w
        return None

    def _nutrition_section(self, state, score, severity, jtype, weight=None):
        """Build the ESPEN-anchored nutrition support section."""
        info = NUTRITION_GUIDANCE.get(state)
        if not info:
            return None
        lang_block = info.get(self.lang, info.get('en', {}))
        content = {
            'state': lang_block.get('label', state),
            'actions': lang_block.get('actions', []),
        }
        if score is not None:
            content['nrs2002'] = score
        if weight is not None:
            content['weight_kg'] = weight
            lo, hi = (35, 40) if state == 'at_risk' else (30, 35)
            content['calculated_targets'] = self._t(
                f'Individualised: ~{int(lo * weight)}-{int(hi * weight)} kcal/day; '
                f'protein ~{weight * 1.2:.0f}-{weight * 1.5:.0f} g/day '
                f'({weight:.0f} kg basis)',
                f'按 {weight:.0f} kg 个体化：约 {int(lo * weight)}-{int(hi * weight)} '
                f'kcal/d；蛋白约 {weight * 1.2:.0f}-{weight * 1.5:.0f} g/d')
        addendum = NUTRITION_SEVERITY_ADDENDA.get(severity)
        if addendum:
            content['severity_note'] = addendum[self.lang]
        type_addendum = NUTRITION_TYPE_ADDENDA.get(jtype)
        if type_addendum:
            content['type_note'] = type_addendum[self.lang]
        return {
            'title': self._t('Nutrition Support', '营养支持'),
            'content': content,
            'icon': '🍎',
            'guidelines': self._gids_to_objs(NUTRITION_GUIDELINE_IDS),
        }


    # ---- v3 agent pipeline ----------------------------------------------
    def _write_audit_record(self, rec):
        """Append a PHI-safe JSONL audit record; never raises into advise()."""
        try:
            import json as _json
            rec = {'timestamp': _datetime.datetime.now().isoformat(timespec='seconds'),
                   **rec}
            with open(self.audit_log, 'a', encoding='utf-8') as f:
                f.write(_json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except Exception:
            pass

    def _agent_run(self, results, labs, diagnosis, dept, severity, jtype,
                   disease_id, diseases_matched, pathway, eff_dept,
                   nutrition_state=None, nrs_score=None):
        """Orchestrate the five agent stages and assemble the audit payload.

        Stage flow (literature-anchored, see module docstring):
        perceive -> route -> reason -> critique -> explain.
        The payload is purely additive: no existing section is altered.
        """
        trace = []
        case = self._agent_perceive(results, labs, diagnosis, dept, trace)
        case['nutrition_state'] = nutrition_state
        conc_level, conc_notes = self._agent_concordance(results, labs, severity)
        case['ai_lab_concordance'] = conc_level
        mode = self._agent_route(case, severity, pathway, conc_level, trace)
        rules = self._agent_collect_rules(results, labs, severity, jtype,
                                          disease_id, eff_dept, pathway, trace,
                                          nutrition_state=nutrition_state,
                                          nrs_score=nrs_score,
                                          clinical_context=bool(
                                              jtype or severity >= 1 or labs))
        self_check = self._agent_critique(case, mode, rules, severity, jtype,
                                          pathway, labs, conc_level, conc_notes,
                                          results, trace)
        trace.append({'stage': 'explain',
                      'detail': f'agent payload assembled: mode={mode}, '
                                f'{len(rules)} rules, priority='
                                f'{self_check["alert_priority"]}'})
        return {
            'pipeline_version': AGENT_PIPELINE_VERSION,
            'mode': mode,
            'case': case,
            'triggered_rules': rules,
            'self_check': self_check,
            'trace': trace,
        }

    def _agent_perceive(self, results, labs, diagnosis, dept, trace):
        """Stage 1 PERCEIVE: build a structured, case-grounded state.

        Accepts both the standard BilinGuard result dict ('screen'/'grade'/...)
        and the legacy compact dict ('binary'/'severity'/'type').
        """
        results = results or {}
        labs = labs or {}
        ai_keys = sorted(k for k in ('screen', 'grade', 'dbil', 'ibil',
                                     'type', 'cp', 'meld', 'binary', 'severity')
                         if results.get(k) is not None)
        screen = results.get('screen')
        if isinstance(screen, dict):
            screen_pos = bool(screen.get('is_jaundice', False))
        elif results.get('binary') is not None:
            screen_pos = bool(results.get('binary'))
        else:
            screen_pos = None
        grade_r = results.get('grade')
        if isinstance(grade_r, dict) and 'grading' in grade_r:
            try:
                ai_grade = int(grade_r['grading'].get('grade_index', 0))
            except (TypeError, ValueError):
                ai_grade = 0
        elif isinstance(results.get('severity'), int):
            ai_grade = int(results['severity'])
        else:
            ai_grade = None
        labs_available = sorted(k for k, v in labs.items() if v is not None)
        labs_missing = [k for k in CRITICAL_LABS if labs.get(k) is None]
        case = {
            'ai_outputs_available': ai_keys,
            'screen_positive': screen_pos,
            'ai_grade': ai_grade,
            'labs_available': labs_available,
            'labs_missing_critical': labs_missing,
            'diagnosis_provided': bool(diagnosis),
            'department_requested': dept,
        }
        trace.append({'stage': 'perceive',
                      'detail': f'{len(ai_keys)} AI outputs, '
                                f'{len(labs_available)} lab values, '
                                f'{len(labs_missing)} critical labs missing, '
                                f'diagnosis={"yes" if diagnosis else "no"}'})
        return case

    def _agent_concordance(self, results, labs, severity):
        """Assess agreement between AI outputs and laboratory values.

        Returns (level, notes): level in 'concordant' / 'minor_discordance' /
        'major_discordance' / 'unassessable'.
        """
        results = results or {}
        labs = labs or {}
        notes = []
        if not labs:
            return 'unassessable', ['no laboratory values provided']

        screen = results.get('screen')
        ai_pos = (bool(screen.get('is_jaundice', False))
                  if isinstance(screen, dict) else
                  (bool(results.get('binary')) if results.get('binary') is not None
                   else None))
        tbil = _num(labs.get('tbil'))
        if ai_pos is not None and tbil is not None:
            if ai_pos and tbil < 34:
                notes.append('screen positive but TBIL < 34 umol/L')
            elif not ai_pos and tbil >= 34:
                notes.append('screen negative but TBIL >= 34 umol/L')

        grade_r = results.get('grade')
        ai_grade = None
        if isinstance(grade_r, dict) and 'grading' in grade_r:
            try:
                ai_grade = int(grade_r['grading'].get('grade_index', 0))
            except (TypeError, ValueError):
                ai_grade = None
        elif isinstance(results.get('severity'), int):
            ai_grade = int(results['severity'])
        if ai_grade is not None and tbil is not None:
            ai_sev = min(ai_grade + 1, 3)
            lab_sev = (3 if tbil > 171 else 2 if tbil >= 85
                       else 1 if tbil >= 34 else 0)
            if abs(ai_sev - lab_sev) >= 2:
                notes.append(f'AI grade implies severity {ai_sev} but labs '
                             f'imply {lab_sev}')
            elif ai_sev != lab_sev:
                notes.append(f'minor grade mismatch (AI {ai_sev} vs labs {lab_sev})')

        if not notes:
            return 'concordant', []
        major = any(('screen' in n or 'implies' in n) for n in notes)
        return ('major_discordance' if major else 'minor_discordance'), notes

    def _agent_route(self, case, severity, pathway, conc_level, trace):
        """Stage 2 ROUTE: select the decision mode for this case."""
        if pathway:
            mode = 'red_flag'
        elif conc_level == 'major_discordance':
            mode = 'conflict'
        elif (case['screen_positive'] or severity >= 1) and case['labs_missing_critical']:
            mode = 'insufficient_data'
        elif case['screen_positive'] or severity >= 1:
            mode = 'routine'
        else:
            mode = 'normal'
        trace.append({'stage': 'route',
                      'detail': f'mode={mode}'
                      + (f' (concordance={conc_level})' if conc_level != 'concordant' else '')})
        return mode

    def _agent_collect_rules(self, results, labs, severity, jtype, disease_id,
                             eff_dept, pathway, trace, nutrition_state=None,
                             nrs_score=None, clinical_context=True):
        """Stage 3 REASON: enumerate triggered rules with GRADE strength."""
        results = results or {}
        labs = labs or {}
        rules = []

        def _add(rule_id, domain, trigger, action, urgency, guideline_ids):
            grade, strength, quality = DOMAIN_GRADE.get(
                domain, ('2C', 'conditional', 'low'))
            rules.append({
                'rule_id': rule_id,
                'domain': domain,
                'trigger': trigger,
                'action': action,
                'urgency': urgency,
                'grade': grade,
                'recommendation_strength': strength,
                'evidence_quality': quality,
                'guideline_ids': [g['gid'] for g in self._gids_to_objs(guideline_ids)],
            })

        # Nutrition rule (v3.1, ESPEN-anchored NRS-2002 pathway)
        if nutrition_state in ('at_risk', 'no_risk') or clinical_context:
            nut_info = NUTRITION_GUIDANCE.get(nutrition_state or 'unscreened')
            nut_label = (nut_info.get(self.lang, nut_info.get('en', {}))
                         .get('label', nutrition_state)) if nut_info else nutrition_state
            if nutrition_state == 'at_risk':
                nut_rule, nut_urg = 'NUTRITION_RISK', 'urgent'
                trig = f'NRS-2002 == {nrs_score} (>= {NUTRITION_RISK_THRESHOLD})'
            elif nutrition_state == 'no_risk':
                nut_rule, nut_urg = 'NUTRITION_RESCREEN', 'routine'
                trig = f'NRS-2002 == {nrs_score} (< {NUTRITION_RISK_THRESHOLD})'
            else:
                nut_rule, nut_urg = 'NUTRITION_SCREEN', 'routine'
                trig = 'NRS-2002 not provided'
            _add(nut_rule, 'nutrition', trig, nut_label, nut_urg,
                 NUTRITION_GUIDELINE_IDS)

        # Severity rule (always fires)
        sev_label = SEVERITY_GUIDANCE.get(
            severity, SEVERITY_GUIDANCE[1])[self.lang].get('label', f'level {severity}')
        _inr_val = _num(labs.get('inr'))
        if labs.get('tbil') is not None:
            sev_src = f'labs TBIL={labs["tbil"]}'
        elif _inr_val is not None and _inr_val > 1.5:
            sev_src = 'INR safety override'
        else:
            sev_src = 'AI grade fallback'
        _add(f'SEVERITY_{severity}', 'severity',
             f'severity == {severity} ({sev_src})', sev_label,
             'routine' if severity <= 1 else ('urgent' if severity == 2 else 'emergency'),
             [g['gid'] for g in self._severity_guidelines(severity)])

        # Jaundice-type rule
        if jtype:
            type_r = results.get('type')
            if isinstance(type_r, (int, float)):
                src = f'AI type_index={int(type_r)}'
            elif isinstance(type_r, dict) and 'type_classification' in type_r:
                src = 'AI type classification'
            else:
                src = 'lab D/T ratio'
            _add(f'TYPE_{jtype}', 'type_classification',
                 f'jaundice_type == {jtype} ({src})',
                 JAUNDICE_TYPES.get(jtype, {}).get(f'{self.lang}_name', jtype),
                 'routine', TYPE_GUIDELINES.get(jtype, []))

        # Disease routing rule
        if disease_id:
            dis = DISEASES.get(disease_id, {})
            kws = dis.get('keywords', [])
            _add(f'DISEASE_{disease_id}', 'disease_routing',
                 f"diagnosis keyword match ({kws[0] if kws else disease_id})",
                 dis.get(f'name_{self.lang}', disease_id),
                 'urgent' if dis.get('pathway') else 'routine',
                 dis.get('guidelines', []))

        # Department routing rule
        if eff_dept:
            info = DEPARTMENTS.get(eff_dept, {})
            _add(f'DEPT_{eff_dept}', 'routing',
                 f'department == {eff_dept}',
                 info.get(f'name_{self.lang}', eff_dept), 'routine',
                 info.get('guidelines', []))

        # Red-flag pathway rule (highest urgency)
        if pathway:
            pid = next((k for k, v in CLINICAL_PATHWAYS.items() if v is pathway), None)
            acts = pathway.get(f'{self.lang}_actions', [])
            action_sum = '; '.join(str(a) for a in acts[:2])
            _add(f'REDFLAG_{pid or "pathway"}', 'red_flag',
                 str(pathway.get(f'{self.lang}_trigger', pid)), action_sum,
                 'emergency', pathway.get('guidelines', []))

        # Child-Pugh / MELD risk stratification rule
        cp = results.get('cp') or {}
        meld = results.get('meld') or {}
        has_cp = isinstance(cp, dict) and 'grading' in cp
        has_meld = isinstance(meld, dict) and 'grading' in meld
        if has_cp or has_meld:
            cgi = mgi = -1
            if has_cp:
                try:
                    cgi = int(cp['grading'].get('grade_index', -1))
                except (TypeError, ValueError):
                    cgi = -1
            if has_meld:
                try:
                    mgi = int(meld['grading'].get('grade_index', -1))
                except (TypeError, ValueError):
                    mgi = -1
            urgency = ('emergency' if cgi == 2 or mgi == 2 else
                       'urgent' if cgi == 1 or mgi == 1 else 'routine')
            _add('CPMELD_RISK', 'risk_stratification',
                 'Child-Pugh / MELD model output present',
                 'liver-reserve risk stratification applied', urgency,
                 CP_MELD_GUIDELINES)

        # Laboratory interpretation rule
        if labs:
            _add('LAB_INTERP', 'lab_interpretation',
                 f'{len(labs)} lab values provided',
                 'ULN-based pattern interpretation', 'routine', [])

        # Triage rule (always fires)
        if severity >= 3:
            tri_urg, tri_act = 'emergency', 'icu/emergency-inpatient level'
        elif severity == 2:
            tri_urg, tri_act = 'urgent', 'inpatient admission'
        elif severity == 1:
            tri_urg, tri_act = 'routine', 'outpatient evaluation'
        else:
            tri_urg, tri_act = 'none', 'routine monitoring only'
        if pathway and severity < 2:
            tri_urg, tri_act = 'urgent', 'prompt 24-48h evaluation (pathway active)'
        _add('TRIAGE', 'triage', f'severity == {severity} -> destination',
             tri_act, tri_urg, [])

        trace.append({'stage': 'reason',
                      'detail': f'{len(rules)} rules triggered '
                                f'({sum(1 for r in rules if r["domain"] == "red_flag")} red-flag)'})
        return rules

    def _agent_critique(self, case, mode, rules, severity, jtype, pathway, labs,
                        conc_level, conc_notes, results, trace):
        """Stage 4 CRITIQUE: pre-output self-check, alert priority, confidence."""
        checks = []

        inr = _num(labs.get('inr'))
        if inr is not None:
            checks.append({
                'check': 'coagulopathy_safety_gate',
                'status': 'FLAG' if inr > 1.5 else 'PASS',
                'detail': f'INR={inr}',
            })

        if pathway:
            checks.append({
                'check': 'severity_pathway_consistency',
                'status': 'PASS' if severity >= 2 else 'FLAG',
                'detail': f'pathway active at severity {severity}',
            })

        if case['labs_missing_critical']:
            checks.append({
                'check': 'critical_data_completeness',
                'status': 'FLAG',
                'detail': 'missing: ' + ', '.join(case['labs_missing_critical']),
            })

        checks.append({
            'check': 'ai_lab_concordance',
            'status': ('PASS' if conc_level == 'concordant'
                       else 'WARN' if conc_level == 'minor_discordance'
                       else 'FLAG' if conc_level == 'major_discordance'
                       else 'WARN'),
            'detail': conc_level + (': ' + '; '.join(conc_notes) if conc_notes else ''),
        })

        if mode == 'red_flag' or severity >= 3:
            priority = 'HIGH'
        elif severity == 2 or mode == 'conflict':
            priority = 'MEDIUM'
        elif severity == 1:
            priority = 'LOW'
        else:
            priority = 'NONE'
        urgency_order = {'emergency': 0, 'urgent': 1, 'routine': 2, 'none': 3}
        ranked = sorted(rules, key=lambda r: urgency_order.get(r['urgency'], 2))
        alerts_ranked = [r['rule_id'] for r in ranked[:3]]

        n_missing = len(case['labs_missing_critical'])
        if (mode in ('conflict', 'insufficient_data') or n_missing >= 2
                or conc_level == 'major_discordance'):
            confidence = 'low'
        elif n_missing == 1 or conc_level in ('minor_discordance', 'unassessable'):
            confidence = 'moderate'
        else:
            confidence = 'high'

        limitations = []
        if case['labs_missing_critical']:
            limitations.append('critical labs missing: '
                               + ', '.join(case['labs_missing_critical']))
        if conc_notes:
            limitations.append('AI-lab discordance: ' + '; '.join(conc_notes))
        if not case['ai_outputs_available']:
            limitations.append('no BilinGuard AI outputs provided')
        stale_cited = sorted({g for r in rules for g in r['guideline_ids']
                              if g in STALE_GUIDELINES})
        if stale_cited:
            limitations.append(f'{len(stale_cited)} cited guideline(s) older '
                               f'than 5 years — verify currency: '
                               + ', '.join(stale_cited[:5]))
        limitations.append('disease inference is keyword-based on free-text diagnosis')
        limitations.append('validated by rule-execution testing on synthetic '
                           'scenarios, not patient-level clinical validation')

        self_check = {
            'checks': checks,
            'alert_priority': priority,
            'alerts_ranked': alerts_ranked,
            'alerts_total': len(rules),
            'confidence': confidence,
            'guidelines_stale_cited': stale_cited,
            'limitations': limitations,
        }
        n_flag = sum(1 for c in checks if c['status'] == 'FLAG')
        trace.append({'stage': 'critique',
                      'detail': f'{len(checks)} checks ({n_flag} flagged), '
                                f'priority={priority}, confidence={confidence}'})
        return self_check


# Module-level convenience for command-line / notebook use
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    demo_results = {
        'screen': {'is_jaundice': True},
        'grade': {'grading': {'grade_index': 1, 'prediction': 'Moderate'}},
        'dbil': {'grading': {'grade_index': 1, 'prediction': 'Mild Elevation'}},
        'type': {'type_classification': {'type_index': 0, 'prediction': 'Hepatocellular'}},
    }
    demo_labs = {'alt': 320, 'ast': 280, 'alp': 130, 'ggt': 90,
                 'tbil': 190, 'dbil': 110, 'ibil': 80, 'inr': 1.6, 'albumin': 32}
    adv = ClinicalAdvisor(lang='cn', department='infectious_hepatology')
    out = adv.advise(demo_results, demo_labs,
                     diagnosis='慢加急性肝衰竭，慢性乙型病毒性肝炎')
    print('SUMMARY:', out['summary'])
    print('SEVERITY:', out['severity'], '| TYPE:', out['jaundice_type'],
          '| DISEASE:', out['disease'])
    for s in out['sections']:
        print('\n===', s['title'], s.get('icon', ''), '===')
        c = s['content']
        if isinstance(c, dict):
            for k, v in c.items():
                if isinstance(v, list):
                    print(f'{k}:')
                    for it in v:
                        print('  -', it)
                else:
                    print(f'{k}: {v}')
        elif isinstance(c, list):
            for it in c:
                print('  -', it)
        else:
            print(c)
    print('\nGUIDELINES:')
    for g in out['guidelines']:
        print(f'  [{g["gid"]}] {g["short"]} ({g["source"]} {g["year"]})')
        print('     ', g['citation'])
    print('\nDISCLAIMER:', out['disclaimer'])
