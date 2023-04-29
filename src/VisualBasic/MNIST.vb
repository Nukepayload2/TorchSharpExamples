' Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
Imports System.IO
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
Imports TorchSharp.Examples.MNIST

''' <summary>
''' Simple MNIST Convolutional model.
''' </summary>
''' <remarks>
''' There are at least two interesting data sets to use with this example:
''' 
''' 1. The classic MNIST set of 60000 images of handwritten digits.
'''
'''     It is available at: http://yann.lecun.com/exdb/mnist/
'''     
''' 2. The 'fashion-mnist' data set, which has the exact same file names and format as MNIST, but is a harder
'''    data set to train on. It's just as large as MNIST, and has the same 60/10 split of training and test
'''    data.
'''    It is available at: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
'''
''' In each case, there are four .gz files to download. Place them in a folder and then point the '_dataLocation'
''' constant below at the folder location.
''' </remarks>
Public NotInheritable Class MNIST
    Private Shared _epochs As Integer = 4
    Private Shared _trainBatchSize As Integer = 64
    Private Shared _testBatchSize As Integer = 128
    Private Shared ReadOnly _logInterval As Integer = 100
    Friend Shared Sub Run(epochs As Integer, timeout As Integer, logdir As String, dataset As String)
        _epochs = epochs

        If String.IsNullOrEmpty(dataset) Then
            dataset = "mnist"
        End If

        Dim device = If(cuda.is_available(), VBHelper.CudaDevice, torch.CPU)

        Console.WriteLine()
        Console.WriteLine($"	Running MNIST with {dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.")
        Console.WriteLine()

        Dim datasetPath As String = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset)

        random.manual_seed(1)

        Dim cwd As String = Environment.CurrentDirectory

        Dim writer = If([String].IsNullOrEmpty(logdir), Nothing, torch.utils.tensorboard.SummaryWriter(logdir, createRunName:=True))

        Dim sourceDir As String = datasetPath
        Dim targetDir As String = Path.Combine(datasetPath, "test_data")

        If Not Directory.Exists(targetDir) Then
            Directory.CreateDirectory(targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir)
        End If

        If device.type = DeviceType.CUDA Then
            _trainBatchSize *= 4
            _testBatchSize *= 4
        End If

        Console.WriteLine($"	Creating the model...")

        Dim model1 As New TorchSharp.Examples.MNIST.Model("model", device)

        Dim normImage = transforms.Normalize(New Double() {0.1307}, New Double() {0.3081}, device:=CType(device, Device))

        Console.WriteLine($"	Preparing training and test data...")
        Console.WriteLine()
        Using train As New MNISTReader(targetDir, "train", _trainBatchSize, device:=device, shuffle:=True, transform:=normImage), test As New MNISTReader(targetDir, "t10k", _testBatchSize, device:=device, transform:=normImage)
            TrainingLoop(dataset, timeout, writer, device, model1, train, test)
        End Using
    End Sub
    Friend Shared Sub TrainingLoop(dataset As String, timeout As Integer, writer As TorchSharp.Modules.SummaryWriter, device As Device, model1 As [Module](Of Tensor, Tensor), train As MNISTReader, test As MNISTReader)
        Dim optimizer = optim.Adam(model1.parameters())

        Dim scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.7)

        Dim totalTime As New Stopwatch
        totalTime.Start()

        For epoch = 1 To _epochs
            MNIST.Train(model1, optimizer, NLLLoss(reduction:=Reduction.Mean), device, train, epoch, train.BatchSize, train.Size)
            MNIST.Test(model1, NLLLoss(reduction:=nn.Reduction.Sum), writer, device, test, epoch, test.Size)

            Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(False)}")

            If totalTime.Elapsed.TotalSeconds > timeout Then
                Exit For
            End If
        Next

        totalTime.[Stop]()
        Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.")

        Console.WriteLine("Saving model to '{0}'", dataset & ".model.bin")
        model1.Save(dataset & ".model.bin")
    End Sub
    Private Shared Sub Train(
        model1 As [Module](Of Tensor, Tensor),
        optimizer As optim.Optimizer,
        loss1 As Loss(Of Tensor, Tensor, Tensor),
        device As Device,
        dataLoader As IEnumerable(Of (Tensor, Tensor)),
        epoch As Integer,
        batchSize1 As Long,
        size1 As Integer)
        model1.train()

        Dim batchId As Integer = 1

        Console.WriteLine($"Epoch: {epoch}...")

        For Each x In dataLoader
            Dim data = x.Item1, target = x.Item2
            Using d = torch.NewDisposeScope()
                optimizer.zero_grad()

                Dim prediction = model1.forward(data)
                Dim output = loss1.forward(prediction, target)

                output.backward()

                optimizer.[step]()

                If batchId Mod _logInterval = 0 Then
                    Console.WriteLine($"
Train: epoch {epoch} [{batchId * batchSize1} / {size1}] Loss: {output.ToSingle():F4}")
                End If

                batchId += 1
            End Using
        Next

    End Sub
    Private Shared Sub Test(
        model1 As [Module](Of Tensor, Tensor),
        loss1 As Loss(Of Tensor, Tensor, Tensor),
        writer As TorchSharp.Modules.SummaryWriter,
        device As Device,
        dataLoader As IEnumerable(Of (Tensor, Tensor)),
        epoch As Integer,
        size1 As Integer)
        model1.eval()

        Dim testLoss As Double = 0
        Dim correct As Integer = 0

        For Each x In dataLoader
            Dim data = x.Item1, target = x.Item2
            Using d = torch.NewDisposeScope()
                Dim prediction = model1.forward(data)
                Dim output = loss1.forward(prediction, target)
                testLoss += output.ToSingle()

                correct += prediction.argmax(1).eq(target).sum().ToInt32()
            End Using
        Next

        Console.WriteLine($"Size: {size1}, Total: {size1}")

        Console.WriteLine($"
Test set: Average loss {(testLoss / size1):F4} | Accuracy {(CDbl(correct) / size1):P2}")

        If writer IsNot Nothing Then
            writer.add_scalar("MNIST/loss", CSng((testLoss / size1)), epoch)
            writer.add_scalar("MNIST/accuracy", CSng(correct) / size1, epoch)
        End If
    End Sub
End Class
