import os
import io
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_report(model_results, demographics_data, gradcam_image_array):
    """
    Generate a comprehensive PDF report with patient demographics, model predictions, 
    Grad-CAM visualization, and AI-generated insights.
    
    Args:
        model_results (dict): Dictionary with 'label', 'prob_cancer', 'prob_non_cancer'
        demographics_data (dict): Dictionary with demographic information
        gradcam_image_array (numpy.ndarray): Grad-CAM overlay image as numpy array
    
    Returns:
        str: Path to the generated PDF file
    """
    try:
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        client = genai.Client(api_key=api_key)
        
        # Generate AI insights using Gemini
        print("Generating AI insights...")
        prompt = f"""You are generating content for a professional medical screening report. Write ONLY the medical content without any conversational phrases, introductions, or meta-commentary.

CRITICAL INSTRUCTION: Do NOT use the words "AI", "artificial intelligence", "AI-generated", "AI-assisted", "AI model", "machine learning", or any similar terminology anywhere in the generated text. Write as if this is a standard clinical screening report from a medical imaging analysis system.

Patient Demographics:
- Age: {demographics_data.get('age', 'N/A')}
- Gender: {demographics_data.get('gender', 'N/A')}
- Smoking Habit: {demographics_data.get('smoking', 'N/A')}
- Alcohol Consumption: {demographics_data.get('alcohol', 'N/A')}
- Tobacco/Betel Nut Chewing: {demographics_data.get('tobacco_chewing', 'N/A')}

Screening Results:
- Classification: {model_results['label']}
- Cancer Probability: {model_results['prob_cancer']:.2%}
- Non-Cancer Probability: {model_results['prob_non_cancer']:.2%}

Generate a professional medical report section following this EXACT structure. Use terms like "screening analysis", "diagnostic imaging assessment", "the screening indicates", "analysis suggests", "imaging findings" instead of any AI-related terms:

RISK ASSESSMENT

[Write 2-3 sentences analyzing the screening prediction in context of the patient's demographic risk factors. Use phrases like "The screening analysis indicates...", "Diagnostic assessment reveals...", "Imaging findings suggest...". Mention specific risk factors present (age, smoking, alcohol, tobacco use) and their contribution to oral cancer risk. Be direct and clinical.]

CLINICAL RECOMMENDATIONS

• [First recommendation - should emphasize immediate professional evaluation by appropriate specialists]
• [Second recommendation - should address lifestyle modifications related to smoking/tobacco if applicable]
• [Third recommendation - should address alcohol consumption if applicable]
• [Fourth recommendation - should cover preventive measures and follow-up]

PROFESSIONAL CONSULTATION

[Write 2-3 sentences emphasizing that this screening is supplementary and that professional medical evaluation is essential. Use phrases like "This screening assessment", "This preliminary analysis", "These imaging findings". Explain that only qualified healthcare providers can provide definitive diagnosis and treatment plans. Do NOT use the word "AI".]

Example format (DO NOT copy this content, generate based on actual patient data):

RISK ASSESSMENT

The screening analysis indicates a moderate probability of oral malignancy requiring further investigation. The patient's age of 52 years combined with a 20-year smoking history presents established risk factors for oral cavity cancers. Heavy alcohol consumption further compounds the baseline risk profile.

CLINICAL RECOMMENDATIONS

• Schedule an immediate consultation with an oral and maxillofacial surgeon or head and neck specialist for comprehensive oral cavity examination and possible biopsy of suspicious lesions.
• Initiate smoking cessation program immediately; evidence shows that discontinuation significantly reduces progression risk even in high-risk patients.
• Reduce alcohol intake to minimal levels; synergistic effects between alcohol and tobacco substantially elevate carcinogenic potential.
• Implement routine oral self-examination protocols and schedule follow-up screenings every 3-6 months regardless of current findings.

PROFESSIONAL CONSULTATION

This screening assessment serves as a preliminary diagnostic tool and does not constitute a definitive medical diagnosis. A comprehensive evaluation by a licensed healthcare provider specializing in oral pathology is essential for accurate diagnosis and treatment planning. Only through direct clinical examination, histopathological analysis, and professional medical judgment can appropriate therapeutic interventions be determined."""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        ai_insights = response.text
        
        # Format AI insights for better PDF rendering
        # Make section headers bold and add proper spacing
        ai_insights = ai_insights.replace('RISK ASSESSMENT', '<b>RISK ASSESSMENT</b>')
        ai_insights = ai_insights.replace('CLINICAL RECOMMENDATIONS', '<br/><br/><b>CLINICAL RECOMMENDATIONS</b>')
        ai_insights = ai_insights.replace('PROFESSIONAL CONSULTATION', '<br/><br/><b>PROFESSIONAL CONSULTATION</b>')
        
        # Create PDF
        print("Creating PDF report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"oral_cancer_report_{timestamp}.pdf"
        pdf_path = os.path.join("working", pdf_filename)
        
        # Ensure working directory exists
        os.makedirs("working", exist_ok=True)
        
        # Create document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                               topMargin=0.5*inch, bottomMargin=0.5*inch,
                               leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        
        # Title
        title = Paragraph("Oral Cancer Screening Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        metadata = Paragraph(f"<i>Generated on: {report_date}</i>", body_style)
        elements.append(metadata)
        elements.append(Spacer(1, 0.3*inch))
        
        # Patient Demographics Section
        demo_heading = Paragraph("Patient Demographics", heading_style)
        elements.append(demo_heading)
        
        demo_data = [
            ['Age:', str(demographics_data.get('age', 'N/A'))],
            ['Gender:', str(demographics_data.get('gender', 'N/A'))],
            ['Smoking Habit:', str(demographics_data.get('smoking', 'N/A'))],
            ['Alcohol Consumption:', str(demographics_data.get('alcohol', 'N/A'))],
            ['Tobacco/Betel Nut Chewing:', str(demographics_data.get('tobacco_chewing', 'N/A'))]
        ]
        
        demo_table = Table(demo_data, colWidths=[2.5*inch, 3.5*inch])
        demo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(demo_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # AI Model Prediction Section
        pred_heading = Paragraph("AI Model Prediction", heading_style)
        elements.append(pred_heading)
        
        # Determine color based on prediction
        pred_color = colors.HexColor('#e74c3c') if model_results['label'] == 'CANCER' else colors.HexColor('#27ae60')
        
        pred_data = [
            ['Classification:', model_results['label']],
            ['Cancer Probability:', f"{model_results['prob_cancer']:.2%}"],
            ['Non-Cancer Probability:', f"{model_results['prob_non_cancer']:.2%}"],
            ['Confidence:', f"{max(model_results['prob_cancer'], model_results['prob_non_cancer']):.2%}"]
        ]
        
        pred_table = Table(pred_data, colWidths=[2.5*inch, 3.5*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('BACKGROUND', (1, 0), (1, 0), pred_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(pred_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Grad-CAM Visualization
        vis_heading = Paragraph("Grad-CAM Visualization", heading_style)
        elements.append(vis_heading)
        
        # Save Grad-CAM image temporarily
        temp_img_path = os.path.join("working", f"temp_gradcam_{timestamp}.png")
        if gradcam_image_array is not None:
            plt.imsave(temp_img_path, gradcam_image_array)
            
            # Add image to PDF
            img = RLImage(temp_img_path, width=5*inch, height=1.67*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.1*inch))
            
            caption = Paragraph(
                "<i>The heatmap shows areas of the image that most influenced the AI's prediction.</i>",
                body_style
            )
            elements.append(caption)
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Clinical Assessment Section
        insights_heading = Paragraph("Clinical Risk Assessment & Recommendations", heading_style)
        elements.append(insights_heading)
        
        # Format the AI insights text properly
        formatted_insights = ai_insights.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
        insights_text = Paragraph(formatted_insights, body_style)
        elements.append(insights_text)
        elements.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['BodyText'],
            fontSize=9,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=TA_JUSTIFY,
            borderWidth=1,
            borderColor=colors.HexColor('#bdc3c7'),
            borderPadding=10,
            backColor=colors.HexColor('#f8f9fa')
        )
        
        disclaimer_text = """
        <b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI system and is intended for 
        screening purposes only. It is NOT a definitive diagnosis. The AI model predictions should 
        be reviewed and validated by qualified healthcare professionals. Please consult with a 
        licensed medical practitioner or oral health specialist for proper diagnosis, treatment 
        recommendations, and medical advice. Early detection and professional evaluation are crucial 
        for oral cancer management.
        """
        
        disclaimer = Paragraph(disclaimer_text, disclaimer_style)
        elements.append(disclaimer)
        
        # Build PDF
        doc.build(elements)
        
        # Clean up temporary image
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        print(f"✓ PDF report generated: {pdf_filename}")
        return pdf_path
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise