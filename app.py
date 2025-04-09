import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template

app = Flask(__name__)

class MoELayer(nn.Module):
    def __init__(self, input_dim=10, num_experts=4, hidden_dim=64, top_k=2):
        super().__init__()
        # Create a list of expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_experts)
        ])
        # Gating network to compute expert scores
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # Compute gating scores and convert them to a probability distribution
        scores = F.softmax(self.gate(x), dim=-1)
        # Select the top_k experts based on the gating probabilities
        probs, indices = torch.topk(scores, self.top_k)
        
        # Initialize the output tensor with the same size as the input
        output = torch.zeros_like(x)
        
        # Loop over each selected expert (separate iteration per expert)
        for expert_idx in range(self.top_k):
            expert_outputs = []  # List to store outputs for each batch element
            # Iterate over each element in the batch
            for b in range(x.size(0)):
                # Retrieve the expert index for the current batch element and expert rank
                selected_expert_index = indices[b, expert_idx]
                # Compute the expert's output for the current input
                expert_output = self.experts[selected_expert_index](x[b])
                expert_outputs.append(expert_output)
            # Stack the outputs for the current expert across the batch dimension
            expert_outputs = torch.stack(expert_outputs)
            # Scale the expert output by its corresponding probability weight
            output += expert_outputs * probs[:, expert_idx].unsqueeze(-1)
            
        # Return the aggregated output and a detached copy of the original gating scores
        return output, scores.detach()

# Initialize the Mixture-of-Experts (MoE) model
model = MoELayer()

@app.route('/', methods=['GET', 'POST'])
def moe_demo():
    if request.method == 'POST':
        # Get and parse the input data from the form (expects comma-separated values)
        input_data = list(map(float, request.form['input_data'].split(',')))
        x = torch.tensor(input_data).unsqueeze(0)
        with torch.no_grad():
            # Run the model to get the output tensor and gating probabilities
            output, probs = model(x)
        # Render the HTML template with output and probability values clearly labeled
        return render_template('index.html', 
                               input_val=request.form['input_data'],
                               output=output.squeeze().tolist(),
                               probs=probs.squeeze().tolist())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)