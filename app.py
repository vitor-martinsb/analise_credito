import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import os
from dash_table import DataTable
from dash_table.Format import Group

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from dash.exceptions import PreventUpdate
import base64
import io

import model
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

columns_features = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_EDUCATION_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
data_loader = model.DataLoader("data/input/",columns = columns_features)
data_train, data_test, target = data_loader.get_data()
feature_engineering = model.FeatureEngineering(data_train, data_test, target)

model_filename = 'predict_probability.pkl'
predict_probability = joblib.load(model_filename)

df = pd.DataFrame(columns=columns_features)

# Função para verificar se as colunas estão presentes
def check_columns(df):
    columns_features = ['SK_ID_CURR','NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                        'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
                        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    missing_columns = [col for col in columns_features if col not in df.columns]
    return missing_columns

# Criar o aplicativo Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Layout da aplicação
app.layout = html.Div([
    # Navbar com gradiente de laranja para roxo
    dbc.Navbar(
        [
            dbc.Col(html.H1("Análise de Crédito", style={'fontFamily': 'Verdana', 'fontWeight': 'bold',
                    'color': 'white', 'textAlign': 'center', 'lineHeight': '50px', 'fontSize': '30px'})),
            dbc.Col(html.Img(
                src="https://www.itau.com.br/media/dam/m/5f03a85a73812bd2/webimage-Header_Logo-Itau.png",
                width="50px",
                height='auto',
                style={'float': 'right','marginRight' : '50px'}
            )),
        ],
        style={'width': '100%',
               'backgroundColor': 'transparent',
               'background': 'linear-gradient(to right, rgba(0,51,153,255), rgba(236,112,0,255))'},
        dark=True,
    ),

        # Upload de arquivo CSV
    dcc.Upload(
        id='upload-data-entrada',
        children=html.Div([
            'Arraste ou Selecione o arquivo ',
            html.A('.csv para análise', style={
                'color': 'rgba(236,112,0,255))', 'fontFamily': 'Verdana', 'fontWeight': 'bold'})
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '0.1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px auto',
            'marginBottom': '20px',
            'marginTop': '20px',
            'color': '#f47d30',
            'cursor': 'pointer',
            'fontFamily': 'Verdana'
        },
        multiple=False
    ),
    
    # Saída do upload
    html.Div(id='output-data-upload'),

    # Botão para executar a análise
    dbc.Row(
        dbc.Col(
            dbc.Button("Executar Análise", id='btn-executar-analise', color="warning", 
                       style={'display': 'none', 'backgroundColor': '#ef761b','fontFamily': 'Verdana'}),
                        width={'size': 6, 'offset': 3},  # Centralizar o botão na grade
                        style={'textAlign': 'center'}  # Centralizar o conteúdo da coluna
        ), style={'margin': 0, 'padding': 0, 'marginTop': 0, 'marginBottom': 0},
    ),

    # Tabela para mostrar os resultados da análise
    dbc.Row(
        dbc.Col(
            DataTable(
                id='data-table',
                editable=False,
                style_table={
                    'backgroundColor': '#2d2d2d',  # Cor de fundo para o tema escuro
                    'color': 'white',  # Cor do texto para o tema escuro
                    'textAlign': 'center',
                    'fontFamily': 'Verdana'
                },
                style_header={
                    'color': 'white',
                    'backgroundColor': '#ef761b',  # Cor de fundo do cabeçalho para o tema escuro
                    'color': 'white',
                    'textAlign': 'center',
                    'fontWeight': 'bold',
                    'fontFamily': 'Verdana'   
                },
                style_cell_conditional=[
                    {
                        'if': {'column_id': col},
                        'backgroundColor': '#262626',  # Cor de fundo das células para o tema escuro
                        'color': 'white',  # Cor do texto das células para o tema escuro
                        'textAlign': 'center',
                        'fontFamily': 'Verdana'
                    } for col in ['SK_ID_CURR','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','Probabilidades']
                ],
                page_size=20,
            ),
            style={'textAlign': 'center'}  # Centralizar o conteúdo da coluna
        ),
        style={'margin': 0, 'padding': 0, 'marginTop': 0, 'marginBottom': 0},
    ),


    html.Div(id='output-execucao', style={'textAlign': 'center'})  # Div para exibir o resultado da execução

])

# Callback para verificar as colunas após o upload do arquivo
@app.callback(Output('output-data-upload', 'children'),
              Output('output-data-upload', 'style'),  # Adicionando uma saída para alterar o estilo
              Input('upload-data-entrada', 'contents'),
              State('upload-data-entrada', 'filename'))
def update_output(contents, filename):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Carregar os dados CSV em um DataFrame
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Verificar se as colunas estão presentes
    missing_columns = check_columns(df)

    if missing_columns:
        return html.Div([
            html.Div(f"Colunas ausentes: {', '.join(missing_columns)}"),
            html.Div("Certifique-se de que o arquivo CSV contém todas as colunas necessárias.")
        ]), {'display': 'none','fontFamily': 'Verdana','textAlign': 'center','color':'#1a4186'}  # Esconder a div de output em caso de erro
    else:
        # Salvar o DataFrame como um arquivo CSV
        data_folder = 'data'
        os.makedirs(data_folder, exist_ok=True)  # Cria a pasta 'data' se não existir
        output_filename = os.path.join(data_folder, 'data_input.csv')
        df.to_csv(output_filename, index=False)

        return html.Div([
            html.Div(f"Arquivo '{filename}' carregado com sucesso !"),
            # Adicione aqui qualquer outra lógica que você deseje após o upload bem-sucedido.
        ]), {'display': 'block','fontFamily': 'Verdana','textAlign': 'center','color':'#1a4186'}, # Mostrar a div de output após o sucesso

# Callback para controlar a visibilidade do botão de execução
@app.callback(Output('btn-executar-analise', 'style'),
              Input('output-data-upload', 'style'))
def update_button_visibility(output_style):
    if output_style['display'] == 'block':
        return {'display': 'inline-block', 'marginBottom': '20px','marginTop': '20px'}
    else:
        return {'display': 'none', 'marginBottom': '20px','marginTop': '20px'}

# Callback para executar a análise quando o botão é clicado
@app.callback(
    Output('data-table', 'data'),  # Dados da tabela
    Output('data-table', 'columns'),  # Estrutura da tabela
    Input('btn-executar-analise', 'n_clicks'))
def execute_analysis(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    # Adicione aqui a lógica de execução da análise usando o arquivo 'data_input.csv' que foi carregado
    data_loader = model.DataLoader("data/input/",columns = columns_features)
    data_train, data_test, target = data_loader.get_data()
    feature_engineering = model.FeatureEngineering(data_train, data_test, target)
    features, target = feature_engineering.engineer_features()

    # Leitura do arquivo 'data_input.csv'
    input_filename = 'data/data_input.csv'
    df = pd.read_csv(input_filename, usecols=columns_features)
    probabilities = predict_probability(feature_engineering.transform_new_data(df))

    df = pd.read_csv('data/data_input.csv',usecols=['SK_ID_CURR','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'])
    df['Probabilidades'] = 1 - probabilities
    print(df)
    df.to_parquet('data/output/resultado.parquet')

    columns = [{'name': col, 'id': col, 'editable': False} for col in df.columns]

    # Retorne os dados e a estrutura da tabela
    return df.to_dict('records'), columns

# Executa o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)