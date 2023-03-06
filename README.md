# LungFiborisisClassification
## Implementation details
The basic architecture we used was a InceptionResNetv2. Categorical cross-entropy was used as the loss function and was minimised using the Adam optimisation algorithm. There were three hyperparameters associated with Adam: 1) the learning rate which was initialised at 1×10^(-3 ), 2) the exponential moving average of the gradient (beta1=0.9), and 3) the exponential moving average of the squared gradient (beta2=0.999). The input image size was 1024×1024 pixels. Random initialisation of model weights was used. The training was performed on 4 GPUs with a batch size of 2 until convergence (Training Set). Optimisation of the algorithm hyperparameters was performed based on the most accurate model on the Validation Set. Finally, the best-performing algorithm was evaluated on Testing Set. 

## Usage
```
python main.py
```
