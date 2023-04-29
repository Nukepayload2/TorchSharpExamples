' Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

Imports System.IO
Imports System.Linq
Imports System.Collections.Generic
Imports System.Diagnostics

Imports TorchSharp
Imports TorchSharp.torchvision

Imports TorchSharp.Examples
Imports TorchSharp.Examples.Utils

Imports TorchSharp.torch

Imports TorchSharp.torch.nn
Imports TorchSharp.torch.nn.functional

Imports cuda = TorchSharp.torch.cuda
Imports Examples.Utils

''' <summary>
''' This example is based on the PyTorch tutorial at:
''' 
''' https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''
''' It relies on the WikiText2 dataset, which can be downloaded at:
'''
''' https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
'''
''' After downloading, extract the files using the defaults (Windows only).
''' </summary>
Public NotInheritable Class SequenceToSequence
    ' This path assumes that you're running this on Windows.
    Private Shared ReadOnly _dataLocation As String = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "wikitext-2-v1")
    Private Const emsize As Long = 200
    Private Const nhid As Long = 200
    Private Const nlayers As Long = 2
    Private Const nhead As Long = 2
    Private Const dropout As Double = 0.2
    Private Const batch_size As Integer = 64
    Private Const eval_batch_size As Integer = 32
    Friend Shared Sub Run(epochs As Integer, timeout As Integer, logdir As String)
        torch.random.manual_seed(1)

        Dim cwd As String = Environment.CurrentDirectory

        Dim device = If(cuda.is_available(), VBHelper.CudaDevice, torch.CPU)

        Console.WriteLine()
        Console.WriteLine($"	Running SequenceToSequence on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.")
        Console.WriteLine()

        Console.WriteLine($"	Preparing training and test data...")

        Dim vocab_iter = TorchText.Datasets.WikiText2("train", _dataLocation)
        Dim tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english")

        Dim counter1 As New TorchText.Vocab.Counter(Of String)
        For Each item In vocab_iter
            counter1.update(tokenizer(item))
        Next

        Dim vocab1 As New TorchText.Vocab.Vocab(counter1)

        Dim TempVar = TorchText.Datasets.WikiText2(_dataLocation)
        Dim train_iter = TempVar.Item1
        Dim valid_iter = TempVar.Item2
        Dim test_iter = TempVar.Item3

        Dim train_data = Batchify(ProcessInput(train_iter, tokenizer, vocab1), batch_size).[to](CType(device, Device))
        Dim valid_data = Batchify(ProcessInput(valid_iter, tokenizer, vocab1), eval_batch_size).[to](CType(device, Device))
        Dim test_data = Batchify(ProcessInput(test_iter, tokenizer, vocab1), eval_batch_size).[to](CType(device, Device))

        Dim bptt As Integer = 32

        Dim ntokens = vocab1.Count

        Console.WriteLine($"	Creating the model...")
        Console.WriteLine()

        Dim model = New TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(CType(device, Device))
        Dim loss = CrossEntropyLoss()
        Dim lr As Double = 2.5
        Dim optimizer = torch.optim.SGD(model.parameters(), lr)
        Dim scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.95, last_epoch:=15)

        Dim writer = If([String].IsNullOrEmpty(logdir), Nothing, torch.utils.tensorboard.SummaryWriter(logdir, createRunName:=True))

        Dim totalTime As System.Diagnostics.Stopwatch = New Stopwatch
        totalTime.Start()

        For Each epoch As Integer In Enumerable.Range(1, epochs)
            Dim sw As System.Diagnostics.Stopwatch = New Stopwatch
            sw.Start()

            train(epoch, train_data, model, loss, bptt, ntokens, optimizer)

            Dim val_loss As Double = evaluate(valid_data, model, loss, bptt, ntokens, optimizer)
            sw.[Stop]()

            Console.WriteLine($"
End of epoch: {epoch} | lr: {optimizer.ParamGroups.First().LearningRate:0.00} | time: {sw.Elapsed.TotalSeconds:0.0}s | loss: {val_loss:0.00}")
            scheduler.[step]()

            If writer IsNot Nothing Then
                writer.add_scalar("seq2seq/loss", CSng(val_loss), epoch)
            End If

            If totalTime.Elapsed.TotalSeconds > timeout Then
                Exit For
            End If
        Next

        Dim tst_loss As Double = evaluate(test_data, model, loss, bptt, ntokens, optimizer)
        totalTime.[Stop]()

        Console.WriteLine($"
End of training | time: {totalTime.Elapsed.TotalSeconds:0.0}s | loss: {tst_loss:0.00}")
    End Sub
    Private Shared Sub train(epoch As Integer, train_data As Tensor, model As TransformerModel, criterion As Loss(Of Tensor, Tensor, Tensor), bptt As Integer, ntokens As Integer, optimizer As torch.optim.Optimizer)
        model.train()

        Dim total_loss As Single = 0.0F
        Using d = torch.NewDisposeScope()
            Dim batch As Integer = 0
            Dim log_interval As Integer = 200

            Dim src_mask = model.GenerateSquareSubsequentMask(bptt)

            Dim tdlen = train_data.shape(0)
            Dim i As Integer = 0


            While i < tdlen - 1
                Dim TupleTempVar As (Tensor, Tensor) = GetBatch(train_data, i, bptt)
                Dim data1 = TupleTempVar.Item1
                Dim targets = TupleTempVar.Item2
                optimizer.zero_grad()

                If data1.shape(0) <> bptt Then
                    src_mask = model.GenerateSquareSubsequentMask(data1.shape(0))
                End If
                Using output = model.forward(data1, src_mask)
                    Dim loss = criterion.forward(output.view(-1, ntokens), targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.[step]()

                    total_loss += loss.[to](torch.CPU).AsSingle()
                End Using

                If batch Mod log_interval = 0 AndAlso batch > 0 Then
                    Dim cur_loss = total_loss / log_interval
                    Console.WriteLine($"epoch: {epoch} | batch: {batch} / {tdlen / bptt} | loss: {cur_loss:0.00}")
                    total_loss = 0
                End If

                d.DisposeEverythingBut(src_mask)
                batch += 1
                i += bptt
            End While
        End Using
    End Sub
    Private Shared Function evaluate(eval_data As Tensor, model As TransformerModel, criterion As Loss(Of Tensor, Tensor, Tensor), bptt As Integer, ntokens As Integer, optimizer As torch.optim.Optimizer) As Double
        model.eval()
        Using d = torch.NewDisposeScope()
            Dim src_mask = model.GenerateSquareSubsequentMask(bptt)

            Dim total_loss = 0.0F
            Dim batch As Integer = 0
            Dim i As Integer = 0


            While i < eval_data.shape(0) - 1
                Dim TupleTempVar1 As (Tensor, Tensor) = GetBatch(eval_data, i, bptt)
                Dim data1 = TupleTempVar1.Item1
                Dim targets = TupleTempVar1.Item2
                If data1.shape(0) <> bptt Then
                    src_mask = model.GenerateSquareSubsequentMask(data1.shape(0))
                End If
                Using output = model.forward(data1, src_mask)
                    Dim loss = criterion.forward(output.view(-1, ntokens), targets)
                    total_loss += data1.shape(0) * loss.[to](torch.CPU).AsSingle
                End Using

                data1.Dispose()
                targets.Dispose()

                d.DisposeEverythingBut(src_mask)
                batch += 1
                i += bptt
            End While

            Return total_loss / eval_data.shape(0)
        End Using
    End Function
    Private Shared Function ProcessInput(iter As IEnumerable(Of String), tokenizer As Func(Of String, IEnumerable(Of String)), vocab1 As TorchText.Vocab.Vocab) As Tensor
        Dim data1 As New List(Of Tensor)
        For Each item As String In iter
            Dim itemData As New List(Of Long)
            For Each token As String In tokenizer(item)
                itemData.Add(vocab1(token))
            Next
            data1.Add(VBHelper.GetTensor(itemData.ToArray(), torch.int64))
        Next

        Dim result = torch.cat(data1.Where(Function(t) t.NumberOfElements > 0).ToList(), 0)
        Return result
    End Function
    Private Shared Function Batchify(data1 As Tensor, batch_size As Integer) As Tensor
        Dim nbatch = data1.shape(0) / batch_size
        Using d2 = data1.narrow(0, 0, CLng(nbatch * batch_size)).view(batch_size, -1).Transpose2D()
            Return d2.contiguous()
        End Using
    End Function
    Private Shared Function GetBatch(source As Tensor, index As Integer, bptt As Integer) As (Tensor, Tensor)
        Dim len = Math.Min(bptt, source.shape(0) - 1 - index)
        Dim data1 = source(TensorIndex.Slice(index, index + len))
        Dim target = source(TensorIndex.Slice(index + 1, index + 1 + len)).reshape(-1)
        Return (data1, target)
    End Function

End Class
