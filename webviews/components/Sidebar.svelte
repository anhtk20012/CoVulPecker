<script>
    let time = null;
    let data = false;
    let filename = null;
    let error = false;
    let isLoading = false;
    let warning = false;

    const onCompleteFile = () => {
        vscode.postMessage({ type: 'onCompleteFile' });
        isLoading = true;
        warning = true;
        time = null;
    };

    const onCodeSelection = () => {
        vscode.postMessage({ type: 'onCodeSelection' });
        isLoading = true;
        warning = true;
        time = null;
    };

    window.addEventListener("message", (event) => {
        if (event.data && typeof event.data === "object") {
            isLoading = false;
            warning = false;
            
            const message = event.data;
            switch (message.command) {
                case 'Prediction':
                    error = false;
                    filename = message.filename;
                    data = message.predictions;
                    time = message.time;
                    break;
                    
                case "onError":
                    error = true;
                    data = message.error;
                    break;
            }
        }
    });
</script>
<style>
    #sb-div {
        box-sizing: border-box;
        padding: 10px;
    }

    #sb-head-div {
        margin-bottom: 10px;
    }

    #sb-head-div img{
        width: 200px;
    }

    #sb-info-div {
        margin-bottom: 10px;
    }

    #sb-info-p {
        text-align: justify;
        font-style: italic;
        line-height: 2;
        margin-bottom: 10px;
    }

    #sb-info-b {
        text-align: justify;
        font-size: large;
    }

    #sb-main-div {
        display: flex;
        align-items:center;
        justify-content: space-between;
    }

    .tooltip {
        position: relative;
    }

    .tooltip::before {
        background-color: rgb(206, 200, 200);
        border: 1px solid rgb(102, 99, 99);
        border-radius: 5px;
        color: #444;
        content: attr(data-title);
        display: none;
        font-size: smaller;
        font-weight: lighter;
        font-style: italic;
        padding: 4%;
        position: absolute;
        right:2px;
        left: 2px;
        top: 90%;
        z-index: 1;
    }

    .tooltip:hover{
        transition-duration: 0.4s;
    }

    .tooltip:hover::before {
        display: block;
    }

    #sb-btn {
        font-weight: bold;
        border-radius: 50px;
        margin: 0 5px;
        margin-bottom: 10px;
    }

    @media (max-width: 400px){
        #sb-main-div{
            flex-direction: column;
        }
        #sb-btn {
            margin: 5px 0;
        }
    }

    #sb-btn:hover {
        color: var(--vscode-button-secondaryForeground);
        background: #043f6f;
    }

    #info-loading {
        font-size: large;
        display: inline-block;
    }

    #info-dot {
        display: inline-block;
        margin-bottom: 10px;
    }

    @keyframes dots {
        from { background-color: rgb(0, 117, 185); }
        to { background-color: white; }
    }

    .dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        margin: 0 5px;
        border-radius: 50%;
    }

    .dot:nth-child(1) {
        animation: dots .5s infinite alternate linear;
    }

    .dot:nth-child(2) {
        animation: dots .6s infinite alternate linear;
    }

    .dot:nth-child(3) {
        animation: dots .7s infinite alternate linear;
    }

    #sb-info-warnign-div{
        background-color: #444;
        border: 1px solid rgb(45, 45, 45);
        padding: 5px;
    }

    #info-warning {
        font-weight: bold;
        font-size: small;
        text-transform: uppercase;
        color:rgb(196, 129, 30);
        margin: 5px;
        text-align: center;
    }

    #sb-results-error-div {
        background-color: #444;
        border: 1px solid rgb(45, 45, 45);
        display: flex;
        padding: 5px;
    }

    #sb-results-error-h1 {
        margin: 5px;
    }
    
    #sb-results-error-p {
        align-content: center;
        text-align: justify;
        font-weight: bold;
        font-style: italic;
        margin: 5px;
        color: #ff6b6b;
    }

    #sb-results-p {
        font-size: large;
        font-style: italic;
        padding: 5px;
    }

    #sb-results-p-filename{
        font-size: small;
        font-style: italic;
        font-weight: bold;
        word-break: break-all;
        color: white;
    }

    #sb-results-div {
        background-color: #444;
        border: 1px solid rgb(45, 45, 45);
        padding: 10px;
    }

    hr {
        border: none;
        height: 1px;
        color: rgb(45, 45, 45); 
        background-color: rgb(45, 45, 45);
    }

    #sb-results-p-linenr {
        color: #76c7c0;
        font-weight: bolder;
        font-size: large;
    }
        
    .progress {
        margin-top: 10px;
        padding: 0;
        height: 20px;
        overflow: hidden;
        border-radius: 50px;
        background: #63b3ed;
    }

    .bar {
        position: relative;
        float: left;
        min-width: 1%;
        height: 100%;
    }

    .percent {
        position: absolute;
        top: 50%;
        left: 50%;
        margin: 0;
        font-size: x-small;
        color: white;
        font-style: italic;
        font-weight: bold;
    }

    #sb-results-p-pred {
        margin: 3% 0% 1% 0%;
        color: white;
        text-transform: uppercase;
        font-size: large;
        font-weight: bold;
    }

</style>
<div id="sb-div">
    <div id="sb-head-div">
        <img class="center" alt="Logo Error" src="https://cdn-icons-png.flaticon.com/256/5262/5262530.png"/>
    </div>

    <div id="sb-info-div">
        <p id="sb-info-p">A VSCode extension designed for detecting vulnerabilities in C/C++ source code. Leveraging an advanced deep learning architecture, this tool effectively identifies vulnerable code patterns.</p>
        <b id="sb-info-b">Click for predictions:</b> 
    </div>

    <div id="sb-main-div">
        <button
            class="tooltip"
            data-title="Select the function code that needs to be analyzed from the active text editor."
            id="sb-btn"
            on:click={()=> {onCodeSelection()}}
            >
            Selected Code
        </button>
        <button
            class="tooltip"
            data-title="Click to analyze all the code from the active text editor."
            id="sb-btn"
            on:click={()=> {onCompleteFile()}}
            >
            Full Code
        </button>
    </div> 

    {#if isLoading}
            <p id="info-loading">Loading: </p>
            <div id="info-dot">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        {#if warning}
            <div id="sb-info-warnign-div">
                <p id="info-warning">⚠️ This action may take some time! </p>
            </div>
        {/if}
    {:else}
        {#if data}
            {#if error}
                <div id="sb-results-error-div">
                    <h1 id="sb-results-error-h1">❌ </h1>
                    <p id="sb-results-error-p">{data}</p>
                </div>
            {:else}
                <div id="sb-results-div">
                    <p id="sb-results-p">Analyzing </p> 
                    <p id="sb-results-p-filename"> {filename}</p>
                    <hr>
                    <div>
                        {#each data as section}
                            <p id="sb-results-p-linenr">LINES: {section.range} </p>
                            {#if section.pred == "True"}
                                <p id="sb-results-p-pred" style="color: #ff6b6b;">❌ Vulnerable</p>
                            {:else if section.pred == "False"}
                                <p id="sb-results-p-pred" style="color: #4CAF50;">✅ Safe</p>  
                            {:else}
                                <p id="sb-results-p-pred" style="color: rgb(196, 129, 30);">⚠️ ERROR SERVER</p>  
                            {/if}
                        {/each}
                    </div>
                    <!-- <hr>
                    {#each data as section}
                        {#each section.lines as lines}
                            {#if section.pred == "True"}
                                <p id="sb-results-p-linenr">LINE: {lines[0]} - {lines[1]}</p>
                                <div class="progress">
                                    <div class="bar" style='width:{lines[2]*100}%; background: #f56565;'>
                                        <p class="percent" style="transform: translate(-50%,-50%);">{Math.ceil(lines[2]*100)}%</p>
                                    </div>
                                    <div class="bar" style='width:{(1-lines[2])*100}%; float: right;'>
                                        <p class="percent" style="transform: translate(-50%,-50%);">{Math.ceil((1-lines[2])*100)}%</p>
                                    </div>
                                </div> 
                            {/if}
                        {/each}
                    {/each}  -->
                    <hr>
                    {#each data as section}
                        {#if section.pred == "True"}
                            <div class="progress">
                                <div class="bar" style='width:{section.average*100}%; background: #f56565;'>
                                    <p class="percent" style="transform: translate(-50%,-50%);">{Math.ceil(section.average*100)}%</p>
                                </div>
                                <div class="bar" style='width:{(1-section.average)*100}%; float: right;'>
                                    <p class="percent" style="transform: translate(-50%,-50%);">{Math.ceil((1-section.average)*100)}%</p>
                                </div>
                            </div>
                        {/if}
                    {/each}
                </div>
            {/if}
        {/if}
    {/if}
    {#if time}
    <div>
        <small>Executed in {time} ms.</small>
    </div>
    {/if}
</div>