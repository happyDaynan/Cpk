from flask import Flask, render_template, request, flash
from flask.helpers import url_for
from werkzeug.utils import redirect, secure_filename
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os, pdfkit

# 設定相關路徑
UPLOAD_FOLDER = './static/upload_folder/'
IMAGES_FOLDER = './static/images_folder/'
PDF_FOLDER = './output/pdf_folder/'

# 設定副檔名
ALLOWED_EXTENSIONS = set(['csv', 'xlsx', 'xls'])

#　pdfkit
pdfkit_Options = {
        'page-size': 'A4',
        'dpi':400,
        'encoding': 'UTF-8',
        'enable-local-file-access': None,
        'disable-smart-shrinking' : ''
    }

pdfkit_Config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')


app = Flask(__name__)

app.secret_key = "MyKey"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER


# 判斷是否是允許的檔案型別
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#　處理資料
def processdata(uploadFile_path, dictData):
    

    df = pd.read_csv(uploadFile_path)
    data = df.values

    dictData['num_samples'] = len(data) # 樣本數量
    dictData['sample_mean'] = round(np.mean(data),5) # 樣本平均數
    dictData['sample_std']  = round(np.std(data), 8) # 樣本標準差
    dictData['sample_max']  = np.max(data)
    dictData['sample_min'] = np.min(data)
    dictData['sample_median']  = np.median(data) # 樣本中位數

    # Calculate Cp and Pp
    dictData['Cp'] = round((dictData['USL'] - dictData['LSL'])/(6*np.std(data)), 2)
    dictData['Pp'] = dictData['Cp']

    # Calculate Ca and Ck
    """
        Ca = Ck = (M-X)/(T/2)
        M:規格中心值，X:量測數據平均值，T:規格寬度(USL-LSL)
        Ca = abs((USL+LSL)/2-sample_median)/((USL-LSL)/2)
    """
    dictData['Ca'] = (dictData['target']  - dictData['sample_mean'])/((dictData['USL'] - dictData['LSL'])/2)
    dictData['Ck'] = dictData['Ca']


    # Calculate Cpu（管制上限）
    """
        Cpu（管制上限）= (USL-u)/3σ
        USL:規格上限，u:量測數據平均值， σ：量測數據的標準差
    """
    dictData['Cpu'] = round((dictData['USL']- dictData['sample_mean'])/(3 * dictData['sample_std']), 2)


    # Calculate Cpl（管制下限)
    """
        Cpl（管制下限 ）= (u-LSL)/3σ
        LSL:規格下限，u:量測數據平均值， σ：量測數據的標準差
    """
    dictData['Cpl'] = round((dictData['sample_mean'] - dictData['LSL'] )/(3 * dictData['sample_std']), 2)

    # Calculate Cpk and Ppk
    """
        Cpk = min｛Cpl, Cpu｝
        Cpu, Cpl取兩者中的最小值就是Cpk
        或 Cpk = min((USL-np.mean(data))/(3*np.std(data)), (np.mean(data)-LSL)/(3*np.std(data)))
    """
    dictData['Cpk'] = round(min(dictData['Cpl'] , dictData['Cpu']), 2)
    dictData['Ppk'] = dictData['Cpk']

    dictData['allData'] = data

    return dictData



@app.route('/')
def index():
    return render_template('index.html', title='Cpk')

@app.route('/show/<cpkData>')
def show(cpkData):
    
    # 這邊會變成字串
    return render_template('show.html', cpkData=cpkData)


@app.route('/Upload',methods=["POST"])
def Upload():
    
    if request.method == 'POST':
        file = request.files['csv_flies']
        print(file)

        dictData = {
            "target" : float(request.form.get('inptarget')),
            "LSL" : float(request.form.get('inpLSL')),
            "USL" : float(request.form.get('inpUSL')),
        }
  
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #print (filename)

            # 上傳資料夾不存在
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                # 建立資料夾
                os.makedirs(app.config['UPLOAD_FOLDER'])

            uploadFile_path =  os.path.join(app.config['UPLOAD_FOLDER'], filename) 

            file.save(uploadFile_path)

            if not os.path.exists(uploadFile_path):
                flash('請重新上傳檔案')
                return redirect(url_for(index))

            showdata = processdata(uploadFile_path, dictData)
            showdata =  showpdf(showdata)

            return redirect(url_for('show', cpkData = showdata))
            # return redirect(render_template('show.html', cpkData = showdata))
            
            # flash('新增成功')
            # return redirect(url_for('index'))
        else:    
            flash('檢查上傳檔案是否正確')
            return redirect(url_for('index'))



def showpdf(showdata):
    
    # def x and y label 
    _x = np.linspace(showdata['USL'], showdata['LSL'], showdata['num_samples'])
    _y = norm.pdf(_x, loc= showdata['sample_mean'], scale= showdata['sample_std'])

    
    plt.figure(figsize=(20,10), dpi= 400)
    plt.hist(showdata['allData'], color='lightgrey', edgecolor="black", bins=2)
    # plt.plot(x, y, linestyle="--", color="black", label="Theorethical Density ST")
    plt.plot(_x, _y, color="red", label="Within")
    plt.plot(_x, _y, linestyle="--", color="black", label="Overall")
    plt.axvline(showdata['LSL'], linestyle="--", color="red", label= "LSL")
    plt.axvline(showdata['USL'], linestyle="--", color="orange", label= "USL")
    # plt.axvline(target, linestyle="--", color="green", label= "Target")
    plt.ylabel("")
    plt.xticks(fontsize=20)
    plt.yticks([])
    plt.legend(fontsize=20)

    _images_Path = os.path.join(app.config['IMAGES_FOLDER'], 'Cpk.jpg')
    
    if not os.path.exists(app.config['IMAGES_FOLDER']):
        os.mkdir(app.config['IMAGES_FOLDER'])

    if os.path.exists(_images_Path):
        os.remove(_images_Path)

    # 結果存檔
    plt.savefig(_images_Path)

    _pdf_Path = os.path.join(app.config['PDF_FOLDER'], 'Cpk.pdf')

    if not os.path.exists(app.config['PDF_FOLDER']):
        os.mkdir(app.config['PDF_FOLDER'])

    if os.path.exists(_pdf_Path):
        os.remove(_pdf_Path)

    pdf_data = render_template('pdf.html', cpkData = showdata)
    pdfkit.from_string(pdf_data, _pdf_Path, configuration= pdfkit_Config, options= pdfkit_Options)

    return showdata


    


if __name__ == '__main__':
    # 設定 debug 模式
    app.run(debug= True)
    