import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

my_model_directory = os.path.join('./', 'models')

y_train = []
with open(os.path.join(my_model_directory, "epochs_train.txt"), "r") as file1:
    for line in file1:
      y_train.append(float(line))


y_valid = []
with open(os.path.join(my_model_directory, "epochs_valid.txt"), "r") as file1:
    for line in file1:
      y_valid.append(float(line))


t = np.linspace(1, len(y_valid), len(y_valid))
x_rev = t[::-1]

valid_scores_mean = np.mean(y_valid)
valid_scores_std = np.std(y_valid)
train_scores_mean = np.mean(y_train)
train_scores_std = np.std(y_train)

ci_valid = []
ci_train = []

for i in range(len(y_valid)):
  ci_valid.append(0.1 * y_valid[i] * valid_scores_std / valid_scores_mean)
  ci_train.append(0.1 * y_train[i] * train_scores_std / train_scores_mean)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y_valid,
                      mode='lines+markers',
                      name='Validation'))
  
fig.add_trace(go.Scatter(x=t, y=y_train,
                      mode='lines+markers',
                      name='Training'))

y_u_valid = []
y_l_valid = []

for i in range(len(y_valid)):
  y_u_valid.append(y_valid[i] + ci_valid[i])
  y_l_valid.append(y_valid[i] - ci_valid[i])
y_l_valid = y_l_valid[::-1]

fig.add_trace(go.Scatter(
  x=np.concatenate([t, x_rev]),
  y=np.concatenate([y_u_valid, y_l_valid]),
  fill='toself',
  fillcolor='rgba(0,176,246,0.2)',
  line_color='rgba(255,255,255,0)', 
))

y_u_train = []
y_l_train = []

for i in range(len(y_train)):
  y_u_train.append(y_train[i] + ci_train[i])
  y_l_train.append(y_train[i] - ci_train[i])
y_l_train = y_l_train[::-1]

fig.add_trace(go.Scatter(
  x=np.concatenate([t, x_rev]),
  y=np.concatenate([y_u_train, y_l_train]),
  fill='toself',
  fillcolor='rgba(255,0,0,0.2)',
  line_color='rgba(255,255,255,0)', 
))

fig.update_layout(legend_title_text = "Phase")
#fig.update_yaxes(type="log", title_text="Log(Value)")
fig.update_xaxes(title_text="Epochs")
fig.update_yaxes(title_text="Value")
fig.show()