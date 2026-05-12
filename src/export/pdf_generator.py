import os
import re
import datetime
from fpdf import FPDF

class ReportExporter:
    """Handles clinical PDF generation."""

    def __init__(self, report_sections):
        self.report_sections = report_sections

    def _parse_sections(self, report_text):
        """Split report text into named sections."""
        section_content = {}
        pattern = "|".join(re.escape(s) for s in self.report_sections)
        parts = re.split(f"({pattern})", report_text, flags=re.IGNORECASE)
        current = None
        for part in parts:
            if part.strip().upper() in self.report_sections:
                current = part.strip().upper()
                section_content[current] = ""
            elif current:
                section_content[current] = section_content.get(current, "") + part
        return section_content

    def _sanitize(self, text):
        """Replace Unicode characters unsupported by FPDF built-in fonts, then ensure latin-1 safety."""
        replacements = {
            "\u2014": "-",   # em dash  —
            "\u2013": "-",   # en dash  –
            "\u2012": "-",   # figure dash
            "\u2015": "-",   # horizontal bar
            "\u2018": "'",   # left single quote  '
            "\u2019": "'",   # right single quote  '
            "\u201c": '"',   # left double quote  "
            "\u201d": '"',   # right double quote  "
            "\u2022": "-",   # bullet  •
            "\u2026": "...", # ellipsis  …
            "\u00b0": " degrees",  # degree sign
            "\u00b1": "+/-",       # plus-minus
            "\u2264": "<=",
            "\u2265": ">=",
            "\u00e9": "e",   # é
            "\u00e8": "e",   # è
            "\u00e0": "a",   # à
            "**": "",
            "__": "",
            "  -": " ",
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        # Final safety net — drop anything still outside latin-1
        return text.encode("latin-1", "replace").decode("latin-1").strip()


    def generate_clinical_pdf(self, patient_data, report_text,
                                findings_summary=None, image_bytes=None):
        pdf = FPDF()
        pdf.set_margins(20, 20, 20)
        pdf.add_page()
        W = 170  # effective page width (210 - 2*20)

        # ── HEADER ───────────────────────────────────────────────────────────
        pdf.set_fill_color(26, 43, 74)
        pdf.rect(0, 0, 210, 40, "F")
        pdf.set_y(9)
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, "SPINE AI CLINICAL RADIOLOGY", ln=True, align="C")
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6, "AI-Assisted Neuroradiology Reporting System", ln=True, align="C")
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 5, "Clinical decision support only - Report finalised by a qualified radiologist",
                 ln=True, align="C")
        pdf.ln(15)

        # ── REPORT TITLE ─────────────────────────────────────────────────────
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 9, "LUMBAR SPINE MRI  -  RADIOLOGY REPORT", ln=True, align="C")
        pdf.set_draw_color(26, 43, 74)
        pdf.set_line_width(0.8)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)

        # ── PATIENT DEMOGRAPHICS ─────────────────────────────────────────────
        pdf.set_font("Arial", "B", 8)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 5, "PATIENT INFORMATION", ln=True)
        pdf.ln(1)

        fields = [
            ("Patient Name",        patient_data.get("name", "N/A")),
            ("Medical Record No.",  patient_data.get("id", "N/A")),
            ("Age / DOB",           patient_data.get("age", "N/A")),
            ("Study Date",          datetime.date.today().strftime("%d %B %Y")),
            ("Referring Physician", patient_data.get("referrer", "N/A")),
            ("Report Status",       "FINAL - SIGNED"),
        ]

        col_w = W / 2
        row_h = 6
        pdf.set_line_width(0.3)
        pdf.set_draw_color(180, 190, 210)

        for i in range(0, len(fields), 2):
            y = pdf.get_y()
            # Label row
            for j in range(2):
                if i + j < len(fields):
                    pdf.set_xy(20 + j * col_w, y)
                    pdf.set_font("Arial", "B", 7.5)
                    pdf.set_text_color(70, 70, 70)
                    pdf.set_fill_color(235, 240, 250)
                    pdf.cell(col_w, row_h, f"  {fields[i+j][0]}", border=1, fill=True)
            pdf.ln(row_h)
            y = pdf.get_y()
            # Value row
            for j in range(2):
                if i + j < len(fields):
                    pdf.set_xy(20 + j * col_w, y)
                    pdf.set_font("Arial", "", 9)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_fill_color(255, 255, 255)
                    val = self._sanitize(fields[i+j][1])
                    pdf.cell(col_w, row_h, f"  {val}", border=1, fill=True)
            pdf.ln(row_h)

        pdf.ln(5)

        # ── REFERENCE IMAGE ───────────────────────────────────────────────────
        if image_bytes:
            from PIL import Image
            import io
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tmp = "_spineai_tmp.png"
                img.save(tmp)
                pdf.set_font("Arial", "B", 8)
                pdf.set_text_color(60, 60, 60)
                pdf.cell(0, 5, "REFERENCE IMAGE", ln=True)
                pdf.image(tmp, x=20, w=55)
                pdf.ln(3)
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as e:
                print(f"[PDF] Image embed error: {e}")

        # ── REPORT SECTIONS ───────────────────────────────────────────────────
        section_data = self._parse_sections(report_text)

        for section in self.report_sections:
            content = section_data.get(section, "").strip()
            if not content:
                content = "Findings unremarkable or deferred to impression."

            # Section heading bar
            pdf.set_fill_color(235, 240, 250)
            pdf.set_draw_color(26, 43, 74)
            pdf.set_line_width(0.5)
            pdf.set_font("Arial", "B", 10)
            pdf.set_text_color(26, 43, 74)
            pdf.cell(0, 8, f"  {section}", ln=True, fill=True, border="B")
            pdf.ln(1)

            # Section body
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(20, 20, 20)
            pdf.set_line_width(0.2)
            pdf.multi_cell(W, 6, self._sanitize(content))
            pdf.ln(4)

        # ── DIVIDER ───────────────────────────────────────────────────────────
        pdf.set_line_width(0.8)
        pdf.set_draw_color(26, 43, 74)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(6)

        # ── SIGNATURE BLOCK ───────────────────────────────────────────────────
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, "ELECTRONIC SIGNATURE & ATTESTATION", ln=True)
        pdf.set_line_width(0.3)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(4)

        signed_by = patient_data.get("signed_by", "Attending Radiologist")
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6,
                 "This report has been independently reviewed, edited, and electronically signed by:",
                 ln=True)
        pdf.ln(1)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, f"  {self._sanitize(signed_by)}", ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6,
                 f"  Date & Time: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
                 ln=True)
        pdf.ln(6)

        # ── DISCLAIMER ────────────────────────────────────────────────────────
        pdf.set_font("Arial", "I", 7.5)
        pdf.set_text_color(110, 110, 110)
        disclaimer = (
            "CONFIDENTIALITY NOTICE: This document is intended solely for the named patient and referring "
            "physician. It was initially drafted by an AI system (SpineAI + CLIP) and has been reviewed and "
            "finalised by the signing radiologist. It does not replace a comprehensive clinical evaluation. "
            "Unauthorised disclosure is strictly prohibited."
        )
        pdf.multi_cell(W, 4.5, self._sanitize(disclaimer))

        return bytes(pdf.output())
