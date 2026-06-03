package com.example.llama

import android.os.Bundle
import android.util.Log
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var ggufTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton

    private var generationJob: Job? = null
    private val messages = mutableListOf<Message>()
    private val messageAdapter = MessageAdapter(messages)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press for simplicity") }

        ggufTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)

        ggufTv.text = "GPT-OSS selective-MoE server: $SERVER_URL"
        userInputEt.hint = "Type and send a message"
        userInputEt.isEnabled = true
        userActionFab.setImageResource(R.drawable.outline_send_24)
        userActionFab.isEnabled = true

        userActionFab.setOnClickListener { handleUserInput() }
    }

    private fun handleUserInput() {
        val userMsg = userInputEt.text.toString()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, "Input message is empty!", Toast.LENGTH_SHORT).show()
            return
        }

        userInputEt.text = null
        userInputEt.isEnabled = false
        userActionFab.isEnabled = false

        messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
        messageAdapter.notifyItemInserted(messages.size - 1)

        messages.add(Message(UUID.randomUUID().toString(), "...", false))
        val assistantIndex = messages.size - 1
        messageAdapter.notifyItemInserted(assistantIndex)
        messagesRv.scrollToPosition(assistantIndex)

        generationJob = lifecycleScope.launch(Dispatchers.IO) {
            val result = runCatching { requestChatCompletion() }
                .getOrElse { error ->
                    Log.e(TAG, "Completion failed", error)
                    "Server error: ${error.message ?: error.javaClass.simpleName}"
                }

            withContext(Dispatchers.Main) {
                messages[assistantIndex] = messages[assistantIndex].copy(content = result)
                messageAdapter.notifyItemChanged(assistantIndex)
                userInputEt.isEnabled = true
                userActionFab.isEnabled = true
            }
        }
    }

    private fun requestChatCompletion(): String {
        val chatMessages = JSONArray().apply {
            put(JSONObject().apply {
                put("role", "system")
                put("content", "Answer directly and briefly. Answer in the user's language.")
            })
            messages.filter { it.content != "..." }.takeLast(8).forEach { message ->
                put(JSONObject().apply {
                    put("role", if (message.isUser) "user" else "assistant")
                    put("content", message.content.trim())
                })
            }
        }
        val payload = JSONObject().apply {
            put("model", "gpt-oss-20b-core-only.gguf")
            put("messages", chatMessages)
            put("temperature", 0.7)
            put("max_tokens", 96)
            put("chat_template_kwargs", JSONObject().apply {
                put("enable_thinking", false)
            })
        }.toString()

        val connection = (URL("$SERVER_URL/v1/chat/completions").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 10_000
            readTimeout = 180_000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        return connection.use { conn ->
            conn.outputStream.use { it.write(payload.toByteArray(Charsets.UTF_8)) }
            val body = if (conn.responseCode in 200..299) {
                conn.inputStream.bufferedReader().use { it.readText() }
            } else {
                val errorBody = conn.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                throw IllegalStateException("HTTP ${conn.responseCode} $errorBody")
            }
            JSONObject(body)
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
                .cleanAssistantText()
        }
    }

    override fun onStop() {
        generationJob?.cancel()
        super.onStop()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val SERVER_URL = "http://127.0.0.1:8080"
    }
}

private fun String.cleanAssistantText(): String {
    val markers = listOf("\nUser:", "<|end|>", "<|return|>", "<|call|>", "<|channel|>", "<|message|>", "<|start|>")
    var cleaned = this
    markers.forEach { marker ->
        val markerIndex = cleaned.indexOf(marker)
        if (markerIndex >= 0) {
            cleaned = cleaned.substring(0, markerIndex)
        }
    }
    return cleaned.trim()
}

private inline fun <T : HttpURLConnection, R> T.use(block: (T) -> R): R {
    try {
        return block(this)
    } finally {
        disconnect()
    }
}
