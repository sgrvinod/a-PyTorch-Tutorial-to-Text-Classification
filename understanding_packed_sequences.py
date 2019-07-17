import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch import nn

# Let's create a batch of sequences of varying length
sequences = torch.FloatTensor([[1, 2, 0, 0, 0, 0],  # length 2
                               [3, 4, 5, 0, 0, 0],  # length 3
                               [5, 6, 0, 0, 0, 0],  # length 2
                               [8, 9, 10, 11, 12, 0]
                               ])  # length 5
seq_lengths = torch.LongTensor([2, 3, 2, 5])
# Since they're of various lengths, they are padded with 0s to a fixed length — this is the only way they can be stored a tensor!

# Apply an RNN over these sequences
rnn = nn.RNN(1, 1, batch_first=True)
rnn_output, _ = rnn(sequences.unsqueeze(2))
rnn_output = rnn_output.squeeze(2)
print(rnn_output)
# As you can see, the RNN computed over the pads, which is wasteful
# Also, you've to manually disregard the RNN outputs at the padded positions for further processing, or loss calculation, or whatever
# Furthermore, if the RNN was bidirectional, the output would be wrong because it will start with the pads in the backward direction

# It's not just with an RNN — any other operation would also compute over the pads
fc = nn.Linear(1, 1)
fc_output = fc(sequences.unsqueeze(2)).squeeze(2)
print(fc_output)

######################################## Packed Sequences ###############################################

# How do we avoid this?
# With PyTorch PackedSequence objects!
packed_sequences = pack_padded_sequence(sequences.unsqueeze(2),
                                        lengths=seq_lengths,
                                        batch_first=True,
                                        enforce_sorted=False)  # The .unsqueeze(2) is simply to add a third dimension which an RNN expects
# This created a PackedSequence object WITHOUT PADS from the padded sequences, upon which a PyTorch RNN can operate directly
rnn_output, _ = rnn(packed_sequences)
# The output of the RNN is a PackedSequence object as well
# So convert it from a PackedSequence back to its padded form
rnn_output, __ = pad_packed_sequence(rnn_output, batch_first=True)
rnn_output = rnn_output.squeeze(2)
print(rnn_output)
# There was no computation at the pads!
# Note that, when it was re-padded, it was re-padded only up to the maximum sequence length (5), and not the original padded length (6)

# What's inside a PackedSequence object?
# The 'sorted_indices' attribute contains the sorting order to reorder the sequences by decreasing lengths — more on the reason for sorting below
print(packed_sequences.sorted_indices)
# The 'data' attribute contains a flattened form of the sorted sequences without the pads
print(packed_sequences.data)
# The 'batch_sizes' attribute notes the effective (non-pad) batch-size at each timestep
print(packed_sequences.batch_sizes)
# The 'unsorted_indices' attribute contains the unsorting order to restore the original order of the sequences
print(packed_sequences.unsorted_indices)

# All of these attributes can also be accessed as a tuple
data, batch_sizes, sorted_indices, unsorted_indices = packed_sequences

# So, RNNs can operate on PackedSequences, and it all magically works without computing over the pads
# What about other operations?
# Since the 'data' attribute is a flattened form without the pads, you can use it for other operations
fc_output = fc(packed_sequences.data)  # and any other operations you want to do
# After everything you need is done, re-pad it into its original form (if required)
fc_output, _ = pad_packed_sequence(PackedSequence(data=fc_output,
                                                  batch_sizes=packed_sequences.batch_sizes,
                                                  sorted_indices=packed_sequences.sorted_indices,
                                                  unsorted_indices=packed_sequences.unsorted_indices),
                                   batch_first=True)
fc_output = fc_output.squeeze(2)
print(fc_output)

######################################## What's really happening here? ###############################################

# A PackedSequence essentially flattens the padded tensor by timestep, keeping only the non-pad units at each timestep

# 1. The sequences are sorted by decreasing sequence lengths, which is the equivalent of:
sorted_lengths, sort_indices = torch.sort(seq_lengths, descending=True)
sorted_sequences = sequences[sort_indices]
# The reason for the sorting is that the non-pads must be concentrated at the top
# This prevents alignment problems when the pads are eliminated
print(sort_indices)
print(packed_sequences.sorted_indices)
print(sorted_sequences)
print(sorted_lengths)

# 2. At each timestep, the effective batch size (excluding the pads) is noted, which is the equivalent of:
effective_batch_sizes = [(i < sorted_lengths).sum().item() for i in range(sorted_sequences.size(1))]
print(effective_batch_sizes)
print(packed_sequences.batch_sizes)

# 3. The sequences are flattened by timestep (excluding the pads), which is the equivalent of:
flattened_sequences = torch.cat(
    [sorted_sequences[:, i][:effective_batch_sizes[i]] for i in range(sorted_sequences.size(1))], dim=0)
print(flattened_sequences)
print(packed_sequences.data.squeeze(1))

# RNNs operate on the sorted sequences only up to the effective batch size (b) at each timestep
# For the next timestep, it takes only the 'b' top outputs from the previous timestep, and so on...
# Please see the tutorial for a visual explanation

# Any other operation, such as a linear layer, can operate directly upon the flattened sequence ('data' attribute) since it doesn't contain any pads

# For something like loss computation over the non-pads, it's really convenient to just do it over the 'data' attribute of a PackedSequence since it will eliminate the pads for you
# I do this in my Image Captioning and Sequence Labeling tutorials, in train.py

# For custom sequential operations, using the effective batch size at each timestep to avoid computation over the pads is very useful
# I do this in my Image Captioning and Sequence Labeling tutorials, search for 'batch_size_t'
