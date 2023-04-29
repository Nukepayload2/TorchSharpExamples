Imports System.IO
Imports System.Reflection
Imports TorchSharp.Examples.Utils

Module Program
    Sub Main(args As String())
        Dim argumentsPath As String = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), "arguments.json")
        Dim argumentParser1 As New ArgumentParser(New FileInfo(argumentsPath), args)

        If argumentParser1.Count = 0 Then
            argumentParser1.UsingMessage("CSharpExamples", "<model-name>")
            Return
        End If

        Dim e As Integer = Nothing
        Dim epochs As Integer = If(argumentParser1.TryGetValueInt("epochs", e), e, 16)
        Dim t As Integer = Nothing
        Dim timeout As Integer = If(argumentParser1.TryGetValueInt("timeout", t), t, 3600)
        Dim ld As String = Nothing
        Dim logdir As String = If(argumentParser1.TryGetValueString("logdir", ld), ld, Nothing)

        For idx = 0 To argumentParser1.Count - 1
            Select Case argumentParser1(idx).ToLower()
                Case "mnist", "fashion-mnist"
                    MNIST.Run(epochs, timeout, logdir, argumentParser1(idx).ToLower())

                Case "fgsm", "fashion-fgsm"
                    AdversarialExampleGeneration.Run(epochs, timeout, logdir, argumentParser1(idx).ToLower())

                Case "alexnet", "resnet", "mobilenet", "resnet18", "resnet34", "resnet50", "vgg11", "vgg13", "vgg16", "vgg19"
                    CIFAR10.Run(epochs, timeout, logdir, argumentParser1(idx))

                Case "text"
                    TextClassification.Run(epochs, timeout, logdir)

                Case "seq2seq"
                    SequenceToSequence.Run(epochs, timeout, logdir)
                Case Else
                    Console.[Error].WriteLine($"Unknown model name: {argumentParser1(idx)}")
            End Select
        Next
    End Sub

End Module
