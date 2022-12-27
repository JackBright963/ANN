# ANN
Implementation of MLP algorithm.
My standard MLP algorithm consists of and is limited to 1 output node and 1 hidden layer, 
but it can operate with any number of neurons, inputs, epochs,
and a step size of any value. I have also made two other models with addition
of momentum and batch learning improvement. All weights and biases are
assigned randomly and will have a value between -n/2 and n/2, where n is the
number of inputs used.

The data is stored as a pre-processed csv file from which the algorithm
reads it. It is standardised and scaled between 0.1 and 0.9.

After cleaning the data, I had 588 rows of data left, which I split as 60%(353
rows), 20%(118 rows) and 20%(117 rows) for training, validation and testing,
respectively.

Unprocessed data is in the file "Raw Data" if needed.

One of the possible improvements to the training method would be randomly
allocating data rows for training, validating and testing. This would maximise
the chances of a given model to go train and adapt to all kinds of different data
and, therefore, make it more flexible.
