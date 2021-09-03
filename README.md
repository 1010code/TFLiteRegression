[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1010code/TFLiteRegression/blob/main/TensorflowRegression.ipynb)

### [文章](https://andy6804tw.github.io/2021/09/02/android-tflite-regression-model-deploy/)

## 將 TFlite 模型部署到 Android 手機
模型一切就緒後緊接著重頭戲就是將模型放到 Android 上讀取並進行預測。首先執行 Android Studio 並開啟一個新專案，其中我們要建立一個 Ktolin 基底的專案。

![](https://andy6804tw.github.io/images/posts/android/2021/img1100902-4.png)

接著打開 `build.gradel(app)` 新增 `tensorflow-lite` 的套件。

```
implementation "org.tensorflow:tensorflow-lite:+"
```

另外為了避免簽署生成 apk 期間壓縮我們的模型，我們需要在該檔案內 `android{ }` 中加入以下描述：

```
aaptOptions {
        noCompress "tflite"
        noCompress "lite"
    }
```

![](https://andy6804tw.github.io/images/posts/android/2021/img1100902-5.png)

接著建立一個 `assets` 資料夾放入稍早所轉換好的 `.tflite` 模型，並將此資料夾放在專案資料夾中 `app -> src -> main` 的位置。

![](https://andy6804tw.github.io/images/posts/android/2021/img1100902-6.png)

我們先處理 layout 畫面，首先建立一個 `EditText` 提供使用者輸入數值，並有一個按鈕送(Button)出並觸發模型預測。最後將預測結果顯示在畫面上。

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <EditText
        android:id="@+id/numberField"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:ems="10"
        android:inputType="number"
        app:layout_constraintBottom_toTopOf="@+id/guideline"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent" />

    <androidx.constraintlayout.widget.Guideline
        android:id="@+id/guideline"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        app:layout_constraintGuide_percent="0.5" />

    <Button
        android:id="@+id/btnPredict"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        android:text="Predict"
        app:layout_constraintEnd_toEndOf="@+id/numberField"
        app:layout_constraintStart_toStartOf="@+id/numberField"
        app:layout_constraintTop_toBottomOf="@+id/numberField" />

    <TextView
        android:id="@+id/txtResult"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="32dp"
        android:fontFamily="sans-serif-black"
        android:textColor="@android:color/black"
        android:textSize="18sp"
        app:layout_constraintEnd_toEndOf="@+id/btnPredict"
        app:layout_constraintStart_toStartOf="@+id/btnPredict"
        app:layout_constraintTop_toBottomOf="@+id/btnPredict" />
</androidx.constraintlayout.widget.ConstraintLayout>
```

接著開啟 `MainActivity` 撰寫主程式，首先建立一個 `initInterpreter()` 函數載入模型並初始化。其中 options 是對模型的一些資源設定，例如我們設定使用 4 個執行緒。以及設定使用 `setUseNNAPI`，Android Neural Networks API (NNAPI) 是一個 Android C API，專門為在邊緣設備上針對機器學習運行計算密集型運算而設計。因為我們在這次範例中使用很簡單的網路層架構，如果是影像辨識專案有使用到許多卷積層相關的 API 那麼該模型是可能無法進行 tflite-android 的 NNAPI 加速的。`loadModelFile()` 函式負責去讀取 `regression.tflite` 並提供模型初始化。第三個函數是 `doInference()` 負責接收使用者輸入的數值，並丟入模型預測。我們可以發現 TFLite 一樣是透過 `interpreter` 進行模型預測，我們需要事先將輸出的變數建立一個空陣列並且使用 `FloatArray`。

```kt
class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private val mModelPath = "regression.tflite"

    private lateinit var resultText : TextView
    private lateinit var editText : EditText
    private lateinit var checkButton : Button
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultText = findViewById(R.id.txtResult)
        editText = findViewById(R.id.numberField)
        checkButton = findViewById(R.id.btnPredict)

        checkButton.setOnClickListener {
            var result = doInference(editText.text.toString())
            runOnUiThread {
                resultText.text = result.toString()
            }
        }

        initInterpreter()
    }

    private fun initInterpreter(){
        val options = Interpreter.Options()
        options.setNumThreads(4)
        options.setUseNNAPI(true)
        interpreter = Interpreter(loadModelFile(assets, mModelPath), options)
    }
    private fun doInference(inputString: String): Float {
        val inputVal = FloatArray(1)
        inputVal[0] = inputString.toFloat()
        val output = Array(1) { FloatArray(1) }
        interpreter.run(inputVal, output)
        return output[0][0]
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
```

<img src="https://andy6804tw.github.io/images/posts/android/2021/img1100902-7.png" width="250px">