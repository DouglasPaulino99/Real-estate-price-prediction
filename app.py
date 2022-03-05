import flask
import pandas as pd
import pickle


with open('model/modelo_simples.pkl', 'rb') as file:
    modelo_importado = pickle.load(file)

with open('model/features_simples.names', 'rb') as file:
    features_name = pickle.load(file)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method =='GET':
        return flask.render_template('home.html')

    if flask.request.method == 'POST':
        #Ver o que digitaram

        user_inputs = {
            'Condo': flask.request.form['Condominio'],
            'Size': flask.request.form['Area'],
            'Rooms': flask.request.form['Quartos'],
            'Suites': flask.request.form['suites']
        }

        df = pd.DataFrame(index=[0], columns=features_name)
        df = df.fillna(value=0)

        for i in user_inputs.items():
             df[i[0]] = i[1]
        df = df.astype(float)

        # Previsão
        y_pred = modelo_importado.predict(df)[0]
        print(y_pred)
        return flask.render_template('home.html', valor_venda=y_pred)

        
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) #Para teste local
    #app.run()

