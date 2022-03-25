#################################################################################
#
#             Project Title:  Training utilities
#             Author:         Alex Boyd
#             Date:           2022-03-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import tqdm
import math

import torch

#################################################################################
#   Function-Class Declaration
#################################################################################

def train(model, epochs, train_data, valid_data, validation_freq=0.1):
  if isinstance(validation_freq, float):
    validation_freq = max(1, int(epochs*validation_freq))

  avg_loss = 0.0
  for epoch in range(1, epochs+1):
    # Training Epoch
    model.train()
    training_pbar = tqdm.tqdm(train_data)
    for i, x in enumerate(training_pbar):
      output = model.train_step(x)
      avg_loss = (avg_loss*i + output["loss"].item()) / (i+1)
      if i % math.ceil(len(train_data) * 0.1) == 0:  # update 10 times over training
        training_pbar.set_description("E={}, Avg [T] Loss: {:.4f}".format(epoch, avg_loss))
    training_pbar.set_description("E={}, Avg [T] Loss: {:.4f}".format(epoch, avg_loss))

    # Validation Epoch
    if (valid_data is not None) and (epoch % validation_freq == 0):
      model.eval()  # disable dropout
      validation_pbar = tqdm.tqdm(valid_data)
      with torch.no_grad():
        for i, x in enumerate(validation_pbar):
          output = model.graded_forward(x)
          avg_loss = (avg_loss*i + output["loss"].item()) / (i+1)
          if i % math.ceil(len(valid_data) * 0.1) == 0:  # update 10 times over validation
            validation_pbar.set_description("E={}, Avg [V] Loss: {:.4f}".format(epoch, avg_loss))
        validation_pbar.set_description("E={}, Avg [V] Loss: {:.4f}".format(epoch, avg_loss))


#################################################################################
#   Main Method
#################################################################################



