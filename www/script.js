// 配置参数
const config = {
    model: "mightykatun/qwen2.5-math:1.5b",
    systemPrompt: "你是一名中学老师，一步一步解答问题", //"一个只会使用中文回答的数学老你是师",
    apiUrl: "http://127.0.0.1:5001/generate"
};
var main_url = "http://127.0.0.1:5002";
var p2t_url = "http://127.0.0.1:5002";
var model2 = "Qwen2.5-math";
var user_id = "123456789";
var user_pwd = "123456789";
var rag_url="http://127.0.0.1:5000/rag"
// 历史对话存储
let history = [];


async function generateUserToken(userId, password) {
    const data = userId + ":" + password; // 组合用户信息
    const encoder = new TextEncoder();
    const hashBuffer = await crypto.subtle.digest("SHA-256", encoder.encode(data)); // 计算 SHA-256 哈希

    // 转换为 Base64
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashBase64 = btoa(hashArray.map(byte => String.fromCharCode(byte)).join(""));

    return hashBase64;
}

// 测试示例
var token = generateUserToken(user_id, user_pwd);
console.log(token);


// 添加消息到聊天框
function addMessage(imageInput, content, isUser = false) {
    const chatBox = document.getElementById('chat-box');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    msgDiv.innerHTML = content+"</br>";  // 修改为 innerHTML 以支持 MathJax 渲染

    if (imageInput.files.length > 0) {
        const file = imageInput.files[0];
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = document.createElement("img");
            img.src = event.target.result;
                 img.style.maxWidth = "600px";  // 最大宽度 200px
        img.style.maxHeight = "600px"; // 最大高度 200px
        img.style.width = "auto";  // 等比缩放
        img.style.height = "auto"; // 等比缩放
        img.style.borderRadius = "5px";
            msgDiv.appendChild(img);
        };
        reader.readAsDataURL(file);
    }


    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // 自动滚动到底部

    // 重新渲染页面中的数学公式
    if (!isUser) {
        MathJax.Hub.Queue(["Typeset", MathJax.Hub, msgDiv]);
    }
    document.getElementById("imageInput").value = '';
    document.getElementById("image-name-container").style.display = "none";
}

// 显示加载状态
function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message loading';
    loadingDiv.id = 'loading';
    loadingDiv.textContent = '思考中...';
    document.getElementById('chat-box').appendChild(loadingDiv);
}

// 隐藏加载状态
function hideLoading() {
    const loadingDiv = document.getElementById('loading');
    if (loadingDiv) loadingDiv.remove();
}

// 处理流式响应
async function handleStreamResponse(response) {
    console.log(response)
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    // //
    // 获取 chat-box 的滚动容器
    const chatBox = document.getElementById("chat-box");

    // 检测用户是否在底部
    function isAtBottom() {
        return chatBox.scrollHeight - chatBox.scrollTop <= chatBox.clientHeight + 10;
    }
// 创建临时消息容器
    const tempMsgDiv = document.createElement('div');
    tempMsgDiv.className = 'message bot-message';
    let res = "";
    document.getElementById('chat-box').appendChild(tempMsgDiv);
    // //
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        // //
        buffer += decoder.decode(value, {stream: true});
        // //
        // 分割完整JSON行
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        // //
        for (const line of lines) {
            if (line.trim() === '') continue;
            try {
                const chunk = JSON.parse(line);
                console.log(chunk)
                if (chunk.message.content && chunk.message.content != "�") {
                    tempMsgDiv.innerHTML += chunk.message.content;
                    res += chunk.message.content;
                    // 记录历史对话
                    tempMsgDiv.scrollIntoView({behavior: 'smooth'});
                    if (isAtBottom()) {
                        chatBox.scrollTop = chatBox.scrollHeight; // 只有在用户在底部时才滚动
                    }
                }
            } catch (err) {
                console.error('解析错误:', err);
            }
        }
    }
    history.push({role: 'assistant', content: res});

    // 渲染消息中的数学公式
    MathJax.Hub.Queue(["Typeset", MathJax.Hub, tempMsgDiv]);
}

var list;
var problems = {};
// 列表示例（你给的部分）
var theorems = {};
// 发送消息
async function sendMessage() {
    const imageInput = document.getElementById("imageInput");
    let image_content = "";

if(imageInput.files.length > 0){
        const file = imageInput.files[0];
        const formData = new FormData();
        formData.append("file", file);
        // fetch("http://127.0.0.1:8080/upload", {
        //     method: "POST",
        //     body: formData
        // })
        // .then(response => response.json())
        // .then(data => {
        //     image_content = data.bold_text;
        //     console.log(123);
        //     // console.log(data.bold_text);
        // })
        // .catch(error => {
        //     console.log("文件上传失败: " + error);
        // });

        var xhr = new XMLHttpRequest();
        xhr.open("POST", p2t_url + "/upload", false); // `false` 表示同步请求
        xhr.setRequestHeader("Authorization", token);
        xhr.send(formData);
        if (xhr.status === 200) {
        try {
            image_content = JSON.parse(xhr.responseText).bold_text; // 提取 `bold_text`
        } catch (error) {
            console.error("解析 JSON 失败:", error);
        }
        } else {
            console.error("文件上传失败，状态码:", xhr.status);
        }

    }

    console.log(image_content);

    list = [[],
    []
];


    const userInput = document.getElementById('user-input');
    const prompt = userInput.value.trim();


    // console.log(12322);
    // 获取用户输入
    // const userInput = document.getElementById('user-input'

    if (!prompt && !image_content) return;
    if (history.length != 0) {
        // 更新历史记录
        config.model = model2;
    }
    if (history.length == 0) {
        // 更新历史记录
        updateHistoryList(prompt);
        // config.model = "Qwen2.5-1.5b-math";
    }
    // 清空输入框
    userInput.value = '';

    // 显示用户消息

    addMessage(imageInput, prompt, true);

    showLoading();

    // 记录历史对话

const url = new URL(rag_url);
url.searchParams.set("prompt", ""+image_content+prompt);

 xhr = new XMLHttpRequest();

xhr.open("GET", url, false); // 同步GET请求
xhr.setRequestHeader("Authorization", token);
xhr.send(); // GET请求无需body
// 发送GET请求并解析JSON
  if (xhr.status >= 200 && xhr.status < 300) {
        try {
            const ul1 = document.getElementById('theorem-list');
  ul1.innerHTML = ''; // 先清空
    const ul2 = document.getElementById('problems-list');
  ul2.innerHTML = ''; // 先清空
            console.log(123213);
            data = JSON.parse(xhr.responseText);
          console.log(data);
          list = data.li;
           theorems = {}
          list[0].forEach(item => {
              const decoded = base64Decode(item.content);
              const key = decoded.slice(0, 4);
              theorems[key] = decoded;
          });
problems = {}
              list[1].forEach(item => {
                  const decoded = base64Decode(item.content);
                  const key = decoded.slice(0, 4);
                  problems[key] = decoded;
              });
              renderTheoremList();
              renderProblemsList();
          }catch (parseError) {
            console.error("JSON解析失败，原始响应：", xhr.responseText);
            console.error("解析错误：", parseError);
        };
  };
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

console.log('Hello');
sleep(4000).then(() => { console.log('World!'); });

history.push({role: 'user', content: "" + image_content+prompt});
    try {
        const response = await fetch(config.apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': token
            },
            body: JSON.stringify({
                model: config.model,
                messages: history,
                stream: true,
                system: config.systemPrompt,
                options: {
                    temperature: 0.7,
                    max_tokens: 500
                },
                // history: history // 发送历史对话
            })
        });

        if (!response.ok) {
            console.log(`请求失败: ${response.status}`);
        }


        hideLoading();
        await handleStreamResponse(response);

        // 记录机器人的回复到历史对话
        // const botResponse = document.querySelector('.message.bot-message').innerHTML;
        // history.push({role: 'assistant', content: botResponse});

    } catch (error) {
        hideLoading();
        addMessage(`错误: ${error.message}`);
    }



    const tempMsgDiv = document.createElement('div');
tempMsgDiv.className = 'message bot-message';

fetch(main_url+'/rel',{
    headers: {
        'Content-Type': 'application/json',
        'Authorization': token
    },
    method: 'POST',
    body: JSON.stringify({
        "question": history,
    })
})
    .then(response => response.json())
    .then(data => {
        if (data.bold_text && Array.isArray(data.bold_text)) {
            const link = document.createElement('a');
            link.innerText = "您接下来可能还想问：";
            tempMsgDiv.appendChild(link);
            data.bold_text.forEach(message => {
                console.log(message);

                const link = document.createElement('a');
                link.href = '#';
                link.innerText = message;
                link.style.display = 'block';
                link.style.color = "RoyalBlue";
                link.style.cursor = 'pointer';

                link.onclick = function(event) {
                    event.preventDefault();
                    const userInput = document.getElementById('user-input');
                    userInput.value = message;
                    document.querySelector('.action-btn').click();
                };

                tempMsgDiv.appendChild(link);
            });
        }
    })

// // 添加到聊天框
document.getElementById('chat-box').appendChild(tempMsgDiv);

}

// 新建聊天
function newChat() {
    document.getElementById('chat-box').innerHTML = '';
    history = [];
    document.getElementById('user-input').value = '';
}

// 更新历史记录
function updateHistoryList(userQuestion) {
const historyList = document.getElementById('history-list');
const historyItem = document.createElement('div');
historyItem.classList.add('history-item');

// 创建文本内容
const textSpan = document.createElement('span');
textSpan.textContent = userQuestion.length > 15 ? userQuestion.substring(0, 15) + '...' : userQuestion;

// 创建垃圾桶按钮
const deleteButton = document.createElement('span');
deleteButton.innerHTML = '🗑'; // 垃圾桶图标（你可以用 FontAwesome 或 SVG 替代）
deleteButton.classList.add('delete-btn');

// 绑定点击事件：点击历史记录进行聊天
historyItem.onclick = () => {
    document.getElementById('user-input').value = userQuestion;
};

// 绑定删除事件：点击垃圾桶图标删除该项
deleteButton.onclick = (event) => {
    newChat();
    event.stopPropagation(); // 防止触发 `historyItem` 的 `onclick`
    historyItem.remove(); // 移除该元素
};

// 组装元素
historyItem.appendChild(textSpan);
historyItem.appendChild(deleteButton);
historyList.appendChild(historyItem);

}

// 加载历史对话
function loadHistory() {
    history.forEach(entry => {
        addMessage(entry.content, entry.role === 'user');
    });
}

// 页面加载时加载历史对话
window.onload = () => {
    loadHistory();
};

// 回车键发送
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});


document.getElementById("imageInput").addEventListener("change", function (event) {
var file = event.target.files[0]; // 获取用户选择的文件
var imageNameContainer = document.getElementById("image-name-container");

if (file) {
    imageNameContainer.textContent = "📎 " + file.name; // 显示文件名
    imageNameContainer.style.display = "block"; // 让元素可见
} else {
    imageNameContainer.style.display = "none"; // 如果没有文件，则隐藏
}
});

// 在script.js中添加
function toggleTheme() {
  document.body.classList.toggle('dark-theme');
  // 保存主题状态
  const isDark = document.body.classList.contains('dark-theme');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}




// 修改后的初始化函数
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  const isDark = savedTheme === 'dark';

  if (isDark) {
    document.body.classList.add('dark-theme');
  }

  // 初始化时也更新参数
}


function openWithToken(liElement) {
    var url = "http://10.124.35.52:8080/pb"; // 目标地址
    const id = liElement.dataset.id;

    // 判断主题，构造 URL
    if((localStorage.getItem('theme') || 'light') == 'light') {
        url = main_url + "/pb?id=" + id + "&is_dark=0";
    } else {
        url = main_url + "/pb?id=" + id + "&is_dark=1";
    }

    // 使用 fetch 发送带 headers 的请求
    console.log(url);
    fetch(url, {
        method: "GET", // 请求方法
        headers: {
            "Authorization": `Bearer ${token}`, // 将 token 放入 headers
        }
    })
    .then(response => response.text())  // 解析为文本（HTML）
    .then(htmlContent => {
        // 打开新标签页并写入 HTML 内容
        const newWindow = window.open(); // 打开新标签页
        newWindow.document.write(htmlContent); // 将返回的 HTML 写入新页面
        newWindow.document.close(); // 关闭文档流
    })
    .catch(error => {
        console.error("Error:", error); // 错误处理
    });
}


// 显示内容详情
function showContent(element, type) {
    const contentArea = document.getElementById('content-area');
    const itemText = element.textContent;

    // 移除之前选中的样式
    document.querySelectorAll('.theorem-section li, .problem-section li').forEach(li => {
        li.style.background = '';
        li.style.color = '';
    });

    // 添加选中样式
    element.style.background = 'var(--button-bg)';
    element.style.color = 'white';

    // 根据类型显示不同内容
    if (type === 'theorem') {
        // 实际应用中这里应该从服务器获取内容
        const content = getTheoremContent(itemText);
        contentArea.innerHTML = content;
    } else if (type === 'problem') {
        // 实际应用中这里应该从服务器获取内容
        const content = getProblemContent(itemText);
        contentArea.innerHTML = content;
    }

    // 渲染数学公式
    MathJax.Hub.Queue(["Typeset", MathJax.Hub, contentArea]);
}

// 获取定理内容（示例）
// function getTheoremContent(name) {
//     const theorems = {
//         "勾股定理": "<p>在直角三角形中，直角边的平方和等于斜边的平方：</p><p>$$ a^2 + b^2 = c^2 $$</p>",
//         "余弦定理": "<p>对于任意三角形，有：</p><p>$$ c^2 = a^2 + b^2 - 2ab\\cos C $$</p>",
//         "韦达定理": "<p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p><p>对于一元二次方程 \( ax^2 + bx + c = 0 \)，根与系数的关系：</p><p>$$ x_1 + x_2 = -\\frac{b}{a} $$</p><p>$$ x_1 \\cdot x_2 = \\frac{c}{a} $$</p>"
//     };
//     return theorems[name] || `<p>${name} 的内容正在加载中...</p>`;
// }
// base64 解码函数（浏览器环境自带 atob，也可以用 Buffer ）
function base64Decode(str) {
  try {
    // 如果是浏览器
    return decodeURIComponent(escape(window.atob(str)));
  } catch (e) {
    // Node.js环境
    return Buffer.from(str, 'base64').toString('utf-8');
  }
}



// 生成列表HTML
function renderTheoremList() {
  const ul = document.getElementById('theorem-list');
  ul.innerHTML = ''; // 先清空
  Object.keys(theorems).forEach(key => {
    const li = document.createElement('li');
    li.textContent = key;
    li.setAttribute('onclick', `showContent(this, 'theorem')`);
    li.style.cursor = 'pointer';
    ul.appendChild(li);
  });
}
function renderProblemsList() {
  const ul = document.getElementById('problems-list');
  ul.innerHTML = ''; // 先清空
  Object.keys(problems).forEach(key => {
    const li = document.createElement('li');
    li.textContent = key;
    li.setAttribute('onclick', `showContent(this, 'problem')`);
    li.style.cursor = 'pointer';
    ul.appendChild(li);
  });
}

// 页面加载后渲染列表

function getTheoremContent(name) {
  return theorems[name].replace(/\$/g, "") || `<p>${name} 的内容正在加载中...</p>`;
}
// 获取题目内容（示例）
function getProblemContent(name) {
    return problems[name].replace(/\$/g, "") || `<p>${name} 的内容正在加载中...</p>`;
}
