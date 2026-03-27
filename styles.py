# ---- SCROLLBAR & WIDGET CSS ---- #
MAIN_STYLE = """
    QWidget { font-family: 'Segoe UI', Arial, sans-serif; background-color: #fefdfa; }
    QScrollBar:horizontal { border: none; background: #fefdfa; height: 8px; margin: 0px; }
    QScrollBar::handle:horizontal { background: #e4d8d8; min-width: 20px; border-radius: 4px; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; }
"""

# ---- RESULT TABLE TEMPLATE ---- #
def get_result_table_html(animal_class, animal_conf, nose_conf, face_d, nose_d):
    return f"""
    <table width='100%' cellspacing='0' cellpadding='6' style='font-family: Segoe UI; border: none; text-align: center;'>
        <tr style='background-color: #fff5f6;'>
            <td width='50%' style='color: #4a4a4a; font-weight: bold; border-bottom: 1px solid #ffccd5; border-right: 1px solid #ffccd5;'>Class</td>
            <td width='50%' style='color: #333;'>{animal_class}</td>
        </tr>
        <tr>
            <td style='color: #4a4a4a; font-weight: bold; border-bottom: 1px solid #ffccd5; border-right: 1px solid #ffccd5;'>Animal Conf</td>
            <td style='color: #333;'>{animal_conf}</td>
        </tr>
        <tr style='background-color: #fff5f6;'>
            <td style='color: #4a4a4a; font-weight: bold; border-bottom: 1px solid #ffccd5; border-right: 1px solid #ffccd5;'>Nose Conf</td>
            <td style='color: #333;'>{nose_conf}</td>
        </tr>
        <tr>
            <td style='color: #4a4a4a; font-weight: bold; border-bottom: 1px solid #ffccd5; border-right: 1px solid #ffccd5;'>Face Dist</td>
            <td style='color: #333;'>{face_d}</td>
        </tr>
        <tr style='background-color: #fff5f6;'>
            <td style='color: #4a4a4a; font-weight: bold; border-right: 1px solid #ffccd5;'>Nose Dist</td>
            <td style='color: #333;'>{nose_d}</td>
        </tr>
    </table>
    """