"""
generate_record_doc.py
Strictly generates the exact 6-PAGE MINI PROJECT EXPERIMENT RECORD
mirroring ML mini project-1.pdf page by page.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx2pdf import convert
import pypdf

def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key in ['sz', 'val', 'color', 'space']:
                if key in edge_data:
                    element.set(qn(f'w:{key}'), str(edge_data[key]))

def set_cell_margins(cell, top=40, bottom=40, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_page_border(section):
    sectPr = section._sectPr
    pgBorders = OxmlElement('w:pgBorders')
    pgBorders.set(qn('w:offsetFrom'), 'page')
    for border_name in ['top', 'left', 'bottom', 'right']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '6')
        border.set(qn('w:space'), '24')
        border.set(qn('w:color'), '000000')
        pgBorders.append(border)
    sectPr.append(pgBorders)

def build_document():
    doc = docx.Document()

    # A4 Dimensions with 0.5 in top/bottom and 0.55 in left/right margins
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.48)
    section.bottom_margin = Inches(0.48)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    
    add_page_border(section)

    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.05
    normal_style.paragraph_format.space_after = Pt(1)
    normal_style.paragraph_format.space_before = Pt(0)

    def add_sec_head(text, space_before=2.5, space_after=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        return p

    def add_sub_head(text, space_before=2, space_after=0.5):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        return p

    def add_p(text, space_before=0, space_after=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.05
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10.5)
        return p

    def add_bullet(text, space_after=0.8):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.05
        r_bullet = p.add_run("•  ")
        r_bullet.bold = False
        r_bullet.font.name = 'Arial'
        r_bullet.font.size = Pt(9.5)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10.5)
        return p

    def add_num(num_str, text, space_after=0.8):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.05
        r_num = p.add_run(f"{num_str}.  ")
        r_num.font.name = 'Times New Roman'
        r_num.font.size = Pt(10.5)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10.5)
        return p

    def add_code(text, space_after=0, indent=0.22):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.02
        p.paragraph_format.left_indent = Inches(indent)
        r = p.add_run(text)
        r.font.name = 'Consolas'
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(15, 15, 15)
        return p

    def add_img(img_path, caption_text, width_in=5.3):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(2)
            p_img.paragraph_format.space_after = Pt(1)
            run = p_img.add_run()
            run.add_picture(img_path, width=Inches(width_in))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(1)
        p_cap.paragraph_format.space_after = Pt(4)
        r_cap = p_cap.add_run(caption_text)
        r_cap.bold = True
        r_cap.font.name = 'Times New Roman'
        r_cap.font.size = Pt(9.5)

    # =========================================================================
    # PAGE 1: Header Box, Aim, Define, Problem Statement, Existing Problems,
    #         Objective, Empathy, Problem Understanding, User Needs, Ideate,
    #         Proposed Ideas (1-5)
    # =========================================================================
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell_left = table.cell(0, 0)
    cell_right = table.cell(0, 1)
    cell_left.width = Inches(1.3)
    cell_right.width = Inches(5.8)
    
    border_spec = {"sz": 6, "val": "single", "color": "000000"}
    set_cell_border(cell_left, top=border_spec, bottom=border_spec, left=border_spec, right=border_spec)
    set_cell_border(cell_right, top=border_spec, bottom=border_spec, left=border_spec, right=border_spec)
    set_cell_margins(cell_left, top=30, bottom=30, left=80, right=80)
    set_cell_margins(cell_right, top=30, bottom=30, left=80, right=80)
    
    p_l1 = cell_left.paragraphs[0]
    p_l1.paragraph_format.space_before = Pt(0)
    p_l1.paragraph_format.space_after = Pt(0)
    p_l1.paragraph_format.line_spacing = 1.0
    r_l1 = p_l1.add_run("EX. NO: 11")
    r_l1.bold = True
    r_l1.font.name = 'Times New Roman'
    r_l1.font.size = Pt(11)
    
    p_l2 = cell_left.add_paragraph()
    p_l2.paragraph_format.space_before = Pt(0)
    p_l2.paragraph_format.space_after = Pt(0)
    p_l2.paragraph_format.line_spacing = 1.0
    r_l2 = p_l2.add_run("DATE:")
    r_l2.bold = True
    r_l2.font.name = 'Times New Roman'
    r_l2.font.size = Pt(11)
    
    p_r = cell_right.paragraphs[0]
    p_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cell_right.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p_r.paragraph_format.space_before = Pt(2)
    p_r.paragraph_format.space_after = Pt(2)
    r_r = p_r.add_run("MINI PROJECT - Agricultural Crop Condition Analysis")
    r_r.bold = True
    r_r.font.name = 'Times New Roman'
    r_r.font.size = Pt(12.5)

    add_sec_head("Aim:", space_before=3, space_after=0.5)
    add_p("To develop a machine learning-based Agricultural Crop Condition Analysis system that groups crops according to environmental conditions and recommends suitable crops based on parameters such as temperature, rainfall, humidity, soil moisture, soil pH, nitrogen, phosphorus, and potassium.")

    add_sec_head("Define:", space_before=2.5, space_after=0.5)
    add_p("In the Define phase, the main problem identified during the empathy phase is clearly stated.")

    add_sub_head("Problem Statement", space_before=2, space_after=0.5)
    add_p('"How might we develop a simple machine learning application that analyzes environmental conditions and groups crops according to their suitability, helping users identify crops that are better suited to specific agricultural conditions?"')

    add_sub_head("Existing Problems", space_before=2, space_after=0.5)
    add_bullet("Farmers may find it difficult to select suitable crops based on multiple environmental parameters.")
    add_bullet("Environmental conditions vary between agricultural regions and fields.")
    add_bullet("Considering temperature, rainfall, humidity, soil moisture, pH, and nutrients simultaneously can be difficult manually.")
    add_bullet("Traditional crop selection may not effectively identify similarities between different crop requirements.")
    add_bullet("There is a need for a simple data-driven crop suitability analysis tool.")

    add_sub_head("Objective", space_before=2, space_after=0.5)
    add_p("The main objectives are:")
    add_num("1", "To collect and analyze crop environmental condition data.")
    add_num("2", "To preprocess and standardize environmental parameters.")
    add_num("3", "To apply K-Means clustering to group crops with similar environmental requirements.")
    add_num("4", "To evaluate the generated clusters using Elbow Method and Silhouette Score.")
    add_num("5", "To analyze user-provided agricultural conditions and identify the closest crop group.")
    add_num("6", "To recommend suitable crops through an interactive Streamlit application.")

    add_sec_head("Empathy:", space_before=2.5, space_after=0.5)
    add_p("The empathy phase focuses on understanding the difficulties faced by farmers and agricultural users in identifying environmentally compatible crops.")

    add_sub_head("Problem Understanding", space_before=2, space_after=0.5)
    add_p("Farmers and agricultural users need an easier way to understand which crops are environmentally compatible with their field conditions. Crop growth is governed by multiple interconnected factors: Temperature, Rainfall, Humidity, Soil Moisture, Soil pH, Nitrogen, Phosphorus, and Potassium.")

    add_sub_head("User Needs", space_before=2, space_after=0.5)
    add_bullet("Users need a simple way to analyze agricultural conditions.")
    add_bullet("The system should require understandable environmental inputs.")
    add_bullet("The analysis should be generated quickly.")
    add_bullet("Crop groups should be presented clearly.")
    add_bullet("Recommended crops should be easy to understand.")
    add_bullet("The system should provide an explanation of why crops are recommended.")
    add_bullet("The interface should be easy to use.")

    add_sec_head("Ideate:", space_before=2.5, space_after=0.5)
    add_p("During the Ideate phase, different possible solutions are considered.")

    add_sub_head("Proposed Ideas", space_before=2, space_after=0.5)
    add_num("1", "Develop a machine learning system for crop condition analysis.")
    add_num("2", "Use environmental and soil parameters as input features.")
    add_num("3", "Apply preprocessing and feature scaling before clustering.")
    add_num("4", "Use K-Means clustering to group crops based on environmental similarity.")
    add_num("5", "Use Elbow Method and Silhouette Score to evaluate cluster quality.")

    # =========================================================================
    # PAGE BREAK 1 -> PAGE 2
    # =========================================================================
    doc.add_page_break()

    # =========================================================================
    # PAGE 2: Proposed Ideas (6-8), Prototype, Major Sections, Code (Beginning)
    # =========================================================================
    add_num("6", "Use PCA for visualizing high-dimensional crop clusters.")
    add_num("7", "Develop a Streamlit-based interactive dashboard.")
    add_num("8", "Provide crop suitability recommendations based on environmental conditions.")

    add_sec_head("Prototype:", space_before=2.5, space_after=0.5)
    add_p("The proposed prototype is an Agricultural Crop Condition Analysis and Crop Suitability Recommendation Dashboard developed using Python, Machine Learning, and Streamlit. The system provides an interactive interface where users can enter eight basic environmental and soil parameters: Temperature, Rainfall, Humidity, Soil Moisture, Soil pH, Nitrogen, Phosphorus, and Potassium.")

    add_p("The dataset contains 220 samples covering 22 crop varieties across five major agricultural categories: Wetland/High Moisture (Rice, Sugarcane, Banana, Jute, Coconut), Semi-Arid/Drought-Tolerant (Millet, Sorghum, Chickpea, Pigeon Pea, Groundnut), Cereals (Wheat, Barley, Maize), Commercial/Cash Crops (Cotton, Soybean, Mustard, Sunflower, Coffee), and Horticultural Crops (Potato, Tomato, Onion, Garlic).")

    add_sub_head("Major Sections", space_before=2, space_after=0.5)
    add_p("1. Home / Overview: Introduces the application, key capabilities, and user workflow.", space_after=0.5)
    add_p("2. Crop Analysis: Input sliders, preset scenarios, and instant suitability recommendations.", space_after=0.5)
    add_p("3. Cluster Analysis: Cluster profiles, radar charts, and interactive 2D/3D PCA cluster visualization.", space_after=0.5)
    add_p("4. Data Exploration: Dataset exploration, statistical summaries, and correlation heatmaps.", space_after=0.5)
    add_p("5. ML Model & Technical Evaluation: Elbow method, Silhouette score evaluation (k=4), and metrics.", space_after=1)

    add_p("This prototype provides an interactive and explainable system to demonstrate how unsupervised machine learning can be applied to precision agriculture.")

    add_sec_head("Code:", space_before=2.5, space_after=1)
    add_code("import streamlit as st")
    add_code("import pandas as pd")
    add_code("import numpy as np")
    add_code("from sklearn.preprocessing import StandardScaler")
    add_code("from sklearn.cluster import KMeans")
    add_code("from sklearn.metrics import silhouette_score")
    add_code("from sklearn.decomposition import PCA")
    add_code("")
    add_code("# Page configuration")
    add_code("st.set_page_config(")
    add_code("    page_title=\"AgriSense AI - Crop Condition Analysis\",")
    add_code("    page_icon=\"🌾\",")
    add_code("    layout=\"wide\"")
    add_code(")")
    add_code("")
    add_code("# Title")
    add_code("st.title(\"🌾 AgriSense AI - Crop Condition Analysis\")")
    add_code("st.write(\"Enter environmental conditions to get crop recommendations.\")")
    add_code("")
    add_code("# Load agricultural dataset")
    add_code("data = pd.read_csv(\"data/crop_dataset.csv\")")
    add_code("")
    add_code("# Feature selection")
    add_code("features = [")
    add_code("    \"temperature\", \"rainfall\", \"humidity\", \"soil_moisture\",")
    add_code("    \"soil_ph\", \"nitrogen\", \"phosphorus\", \"potassium\"")
    add_code("]")
    add_code("X = data[features]")

    # =========================================================================
    # PAGE BREAK 2 -> PAGE 3
    # =========================================================================
    doc.add_page_break()

    # =========================================================================
    # PAGE 3: Code Continued (Scaling, KMeans k=4, Silhouette, PCA, Inputs 1-4)
    # =========================================================================
    add_code("# Preprocess and standardize data")
    add_code("scaler = StandardScaler()")
    add_code("X_scaled = scaler.fit_transform(X)")
    add_code("")
    add_code("# Train K-Means Clustering model (optimal k = 4)")
    add_code("optimal_k = 4")
    add_code("kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)")
    add_code("clusters = kmeans.fit_predict(X_scaled)")
    add_code("data[\"Cluster\"] = clusters")
    add_code("")
    add_code("# Cluster evaluation using Silhouette Score")
    add_code("sil_score = silhouette_score(X_scaled, clusters)")
    add_code("")
    add_code("# Dimensionality reduction with PCA for 2D visualization")
    add_code("pca = PCA(n_components=2)")
    add_code("pca_coords = pca.fit_transform(X_scaled)")
    add_code("data[\"PCA1\"] = pca_coords[:, 0]")
    add_code("data[\"PCA2\"] = pca_coords[:, 1]")
    add_code("")
    add_code("# User environmental inputs interface")
    add_code("st.subheader(\"Enter Agricultural Environmental Conditions\")")
    add_code("")
    add_code("temperature = st.number_input(")
    add_code("    \"Temperature (°C)\",")
    add_code("    min_value=5.0,")
    add_code("    max_value=50.0,")
    add_code("    value=26.5")
    add_code(")")
    add_code("")
    add_code("rainfall = st.number_input(")
    add_code("    \"Rainfall (mm)\",")
    add_code("    min_value=100.0,")
    add_code("    max_value=3500.0,")
    add_code("    value=1800.0")
    add_code(")")
    add_code("")
    add_code("humidity = st.number_input(")
    add_code("    \"Relative Humidity (%)\",")
    add_code("    min_value=15.0,")
    add_code("    max_value=100.0,")
    add_code("    value=80.0")
    add_code(")")
    add_code("")
    add_code("soil_moisture = st.number_input(")
    add_code("    \"Soil Moisture (%)\",")
    add_code("    min_value=10.0,")
    add_code("    max_value=90.0,")
    add_code("    value=65.0")
    add_code(")")

    # =========================================================================
    # PAGE BREAK 3 -> PAGE 4
    # =========================================================================
    doc.add_page_break()

    # =========================================================================
    # PAGE 4: Code Continued (Inputs 5-8, Button, Input Data, Scaling,
    #         Cluster Prediction, Gaussian Suitability Formula)
    # =========================================================================
    add_code("soil_ph = st.number_input(")
    add_code("    \"Soil pH\",")
    add_code("    min_value=4.0,")
    add_code("    max_value=9.5,")
    add_code("    value=6.5")
    add_code(")")
    add_code("")
    add_code("nitrogen = st.number_input(")
    add_code("    \"Nitrogen (N) kg/ha\",")
    add_code("    min_value=10.0,")
    add_code("    max_value=250.0,")
    add_code("    value=120.0")
    add_code(")")
    add_code("")
    add_code("phosphorus = st.number_input(")
    add_code("    \"Phosphorus (P) kg/ha\",")
    add_code("    min_value=5.0,")
    add_code("    max_value=150.0,")
    add_code("    value=50.0")
    add_code(")")
    add_code("")
    add_code("potassium = st.number_input(")
    add_code("    \"Potassium (K) kg/ha\",")
    add_code("    min_value=10.0,")
    add_code("    max_value=250.0,")
    add_code("    value=55.0")
    add_code(")")
    add_code("")
    add_code("# Analyze crop suitability")
    add_code("if st.button(\"Analyze Crop Suitability\"):")
    add_code("    input_data = pd.DataFrame([[")
    add_code("        temperature,")
    add_code("        rainfall,")
    add_code("        humidity,")
    add_code("        soil_moisture,")
    add_code("        soil_ph,")
    add_code("        nitrogen,")
    add_code("        phosphorus,")
    add_code("        potassium")
    add_code("    ]], columns=features)")
    add_code("")
    add_code("    # Standardize input data")
    add_code("    input_scaled = scaler.transform(input_data)")
    add_code("")
    add_code("    # Predict closest cluster")
    add_code("    prediction = kmeans.predict(input_scaled)[0]")
    add_code("")
    add_code("    # Compute suitability scores using Gaussian RBF formula:")
    add_code("    # S = 100 * exp(-d^2 / (2 * sigma^2))")
    add_code("    sigma = 1.0")
    add_code("    crop_scores = []")
    add_code("    for crop in data[\"crop_name\"].unique():")
    add_code("        crop_data = data[data[\"crop_name\"] == crop][features]")
    add_code("        crop_mean = scaler.transform(crop_data).mean(axis=0)")
    add_code("        dist = np.linalg.norm(input_scaled[0] - crop_mean)")
    add_code("        suitability = 100.0 * np.exp(-(dist ** 2) / (2 * (sigma ** 2)))")
    add_code("        crop_scores.append((crop, suitability))")

    # =========================================================================
    # PAGE BREAK 4 -> PAGE 5
    # =========================================================================
    doc.add_page_break()

    # =========================================================================
    # PAGE 5: Code Concluded, App Section, Fig. 1 Screenshot
    # =========================================================================
    add_code("    # Rank and display top recommendations")
    add_code("    crop_scores.sort(key=lambda x: x[1], reverse=True)")
    add_code("    top_crops = crop_scores[:3]")
    add_code("")
    add_code("    # Display result")
    add_code("    st.success(f\"Assigned Cluster Group: Cluster {prediction}\")")
    add_code("    for crop_name, score in top_crops:")
    add_code("        st.metric(label=f\"Recommended: {crop_name}\", value=f\"{score:.1f}%\")")
    add_code("")
    add_code("    st.info(\"Suitability scores computed via Euclidean distance and Gaussian radial basis scoring.\")")

    add_sec_head("App:", space_before=8, space_after=3)
    add_img("assets/screenshots/01_home.png", "Fig. 1: AgriSense AI Crop Condition Analysis Application", width_in=5.4)

    # =========================================================================
    # PAGE BREAK 5 -> PAGE 6
    # =========================================================================
    doc.add_page_break()

    # =========================================================================
    # PAGE 6: Screenshots 2 & 3, Result
    # =========================================================================
    add_img("assets/screenshots/03_recommendation.png", "Fig. 2: Crop Suitability Recommendation Result", width_in=5.2)
    add_img("assets/screenshots/06_ml_evaluation.png", "Fig. 3: Machine Learning Model Evaluation & Cluster Analysis Dashboard", width_in=5.2)

    add_sec_head("Result:", space_before=3, space_after=1.5)
    add_p("The Agricultural Crop Condition Analysis system was successfully developed using Machine Learning and Streamlit. The system analyzes environmental and soil parameters, groups crops using K-Means clustering, evaluates the clusters using Elbow Method and Silhouette Score, and recommends suitable crops based on the entered agricultural conditions. The system was tested with different environmental inputs and successfully generated crop group and suitability recommendations.")

    # Save DOCX
    docx_path = "docs/MINI_PROJECT_EXPERIMENT_RECORD.docx"
    doc.save(docx_path)
    print(f"Saved DOCX to {docx_path}")

    # Convert to PDF
    pdf_path = "docs/MINI_PROJECT_EXPERIMENT_RECORD.pdf"
    print(f"Converting {docx_path} to {pdf_path}...")
    convert(docx_path, pdf_path)
    print(f"Saved PDF to {pdf_path}")

    # Verify Page Count
    reader = pypdf.PdfReader(pdf_path)
    page_count = len(reader.pages)
    print(f"Total PDF pages: {page_count}")
    return docx_path, pdf_path, page_count

if __name__ == "__main__":
    docx_path, pdf_path, page_count = build_document()
    if page_count != 6:
        print(f"WARNING: Expected exactly 6 pages, got {page_count} pages!")
    else:
        print("SUCCESS: Exact 6-page document generated!")
