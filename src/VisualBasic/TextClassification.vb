' Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

Imports System.IO
Imports System.Linq
Imports System.Collections.Generic
Imports System.Diagnostics

Imports TorchSharp

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
''' https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
'''
''' It relies on the AG_NEWS dataset, which can be downloaded in CSV form at:
'''
''' https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv
'''
''' Download the two files, and place them in a folder called "AG_NEWS" in
''' accordance with the file path below (Windows only).
'''
''' </summary>
Public NotInheritable Class TextClassification
    Private Const emsize As Long = 200
    Private Const batch_size As Long = 128
    Private Const eval_batch_size As Long = 128
    Private Const epochs As Integer = 15
    ' This path assumes that you're running this on Windows.
    Private Shared ReadOnly _dataLocation As String = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "AG_NEWS")
    Friend Shared Sub Run(epochs As Integer, timeout As Integer, logdir As String)
        torch.random.manual_seed(1)

        Dim cwd As String = Environment.CurrentDirectory

        Dim device = If(cuda.is_available(), VBHelper.CudaDevice, torch.CPU)

        Console.WriteLine()
        Console.WriteLine($"	Running TextClassification on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.")
        Console.WriteLine()

        Console.WriteLine($"	Preparing training and test data...")
        Using reader = TorchText.Data.AG_NEWSReader.AG_NEWS("train", CType(device, Device), _dataLocation)
            Dim dataloader = reader.Enumerate()

            Dim tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english")

            Dim counter1 As New TorchText.Vocab.Counter(Of String)
            For Each x In dataloader
                Dim label = x.Item1, text = x.Item2

                counter1.update(tokenizer(text))
            Next

            Dim vocab1 As New TorchText.Vocab.Vocab(counter1)


            Console.WriteLine($"	Creating the model...")
            Console.WriteLine()

            Dim model = New TextClassificationModel(vocab1.Count, emsize, 4).to(CType(device, Device))

            Dim loss = CrossEntropyLoss()
            Dim lr As Double = 5.0
            Dim optimizer = torch.optim.SGD(model.parameters(), lr)
            Dim scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.2, last_epoch:=5)

            Dim totalTime As System.Diagnostics.Stopwatch = New Stopwatch
            totalTime.Start()

            For Each epoch As Integer In Enumerable.Range(1, epochs)
                Dim sw As System.Diagnostics.Stopwatch = New Stopwatch
                sw.Start()

                train(epoch, reader.GetBatches(tokenizer, vocab1, batch_size), model, loss, optimizer)

                sw.[Stop]()

                Console.WriteLine($"
End of epoch: {epoch} | lr: {optimizer.ParamGroups.First().LearningRate:0.0000} | time: {sw.Elapsed.TotalSeconds:0.0}s")
                scheduler.[step]()

                If totalTime.Elapsed.TotalSeconds > timeout Then
                    Exit For
                End If
            Next

            totalTime.[Stop]()
            Using test_reader = TorchText.Data.AG_NEWSReader.AG_NEWS("test", CType(device, Device), _dataLocation)
                Dim sw As System.Diagnostics.Stopwatch = New Stopwatch
                sw.Start()

                Dim accuracy As Double = evaluate(test_reader.GetBatches(tokenizer, vocab1, eval_batch_size), model, loss)

                sw.[Stop]()

                Console.WriteLine($"
End of training: test accuracy: {accuracy:0.00} | eval time: {sw.Elapsed.TotalSeconds:0.0}s")
                scheduler.[step]()
            End Using
        End Using
    End Sub
    Private Shared Sub train(epoch As Integer, train_data As IEnumerable(Of (Tensor, Tensor, Tensor)), model As TextClassificationModel, criterion As Loss(Of Tensor, Tensor, Tensor), optimizer As torch.optim.Optimizer)
        model.train()

        Dim total_acc As Double = 0.0
        Dim total_count As Long = 0
        Dim log_interval As Long = 250

        Dim batch As Integer = 0

        Dim batch_count As Integer = train_data.Count()
        Using d = torch.NewDisposeScope()
            For Each x In train_data
                Dim labels = x.Item1, texts = x.Item2, offsets = x.Item3

                optimizer.zero_grad()
                Using predicted_labels = model.forward(texts, offsets)
                    Dim loss = criterion.forward(predicted_labels, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.[step]()
                    Dim tempVar = predicted_labels.argmax(1) = labels
                    total_acc += tempVar.sum().[to](torch.CPU).AsLong()
                    total_count += labels.size(0)
                End Using

                If batch Mod log_interval = 0 AndAlso batch > 0 Then
                    Dim accuracy = total_acc / total_count
                    Console.WriteLine($"epoch: {epoch} | batch: {batch} / {batch_count} | accuracy: {accuracy:0.00}")
                End If
                batch += 1
            Next

        End Using
    End Sub
    Private Shared Function evaluate(test_data As IEnumerable(Of (Tensor, Tensor, Tensor)), model As TextClassificationModel, criterion As Loss(Of Tensor, Tensor, Tensor)) As Double
        model.eval()

        Dim total_acc As Double = 0.0
        Dim total_count As Long = 0
        Using d = torch.NewDisposeScope()
            For Each x In test_data
                Dim labels = x.Item1, texts = x.Item2, offsets = x.Item3
                Using predicted_labels = model.forward(texts, offsets)
                    Dim loss = criterion.forward(predicted_labels, labels)
                    Dim tempVar = predicted_labels.argmax(1) = labels
                    total_acc += tempVar.sum().[to](torch.CPU).AsLong()
                    total_count += labels.size(0)
                End Using
            Next
            Return total_acc / total_count
        End Using
    End Function
End Class
