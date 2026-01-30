
## Project Overview

  

This repository contains two versions of an AI-powered data analysis application built with Streamlit. The main branch features an interactive **Data Analysis Copilot** with real-time agent execution, while V0.1 demonstrates a simpler CSV report generation workflow.

  

## Branch Structure

  

### **main** (V0.2_Data_Analysis_Copilot)

The primary development branch featuring an advanced **interactive Data Analysis Copilot** with a 4-panel interface.

  

**Purpose**: Production-ready copilot for interactive data analysis with conversational AI, live agent execution, and customizable reporting.

  

**Key Features**:

- **4-Panel UI Layout**: Chatbot (top-left), Plan/Code tabs (top-right), Editable data table (bottom-left), Live report (bottom-right)

- **Conversational Chatbot**: OpenAI function calling to detect when users want to create reports vs ask simple questions

- **Plan Generation**: AI generates step-by-step execution plans with tool assignments (web_search, python_repl_tool)

- **Live Agent Execution**: LangChain AgentExecutor with DuckDuckGoSearchResults and PythonREPLTool

- **Interactive Code Editing**: Users can edit report visualization code using streamlit-ace

- **Real-time Data Editing**: Editable dataframe with dynamic row addition

- **LangSmith Integration**: Optional monitoring of agent activities

  

**Architecture Differences from V0.1**:

- Uses actual OpenAI API calls (not mocked responses)

- Agent-based execution with intermediate step logging

- Two-stage workflow: plan generation → execution → report code generation

- Session state management with auto-refresh mechanism

- OpenAI function calling for intent detection (trigger_report_generation tool)

  

### **V0.1_CSV_Analysis_Report_Generation**

Demo branch showcasing a simpler, linear CSV analysis workflow.

  

**Purpose**: Proof-of-concept demonstrating the agent pipeline concept with mocked responses (no live API calls needed).

  

**Key Features**:

- Single-page workflow: Upload → Analyze → Filter → Visualize → Export

- Four AI agents in sequence: Schema Agent → Processing Agent → Plotting Agent → Analysis Agent

- Word document (.docx) export with embedded charts

- Hardcoded responses with time delays to simulate AI processing

- Designed for sensor data (Temperature, Vibration, Pressure, Flow Rate)

  

**Use Case**: Educational demo or starting point for understanding multi-agent workflows without requiring OpenAI API access.

  

## Running the Application

  

### Setup

  

1. Install dependencies:

```bash

pip install -r requirements.txt

```

  

2. Create a `.env` file in the root directory with your OpenAI API key:

```

OPENAI_API_KEY=your_api_key_here

```

  

### Running the App

  

```bash

streamlit run streamlit_app.py

```

  

The app will open in your browser at `http://localhost:8501`.

  

## Architecture (main branch)

  

### Three-Stage Agent Workflow

  

**Stage 1: Conversational Intent Detection** (lines 86-243)

- Chatbot uses OpenAI function calling with `trigger_report_generation` tool

- Detects when user wants to create a report vs asking simple questions

- For simple questions: uses `create_pandas_dataframe_agent` for direct answers

  

**Stage 2: Plan Generation** (lines 145-176)

- AI creates hierarchical execution plan with parent steps and substeps

- Each parent step is assigned a tool: `[Tool: python_repl_tool]` or `[Tool: web_search]`

- Plan is displayed in the "Plan" tab for user review before execution

  

**Stage 3: Agent Execution & Report Code Generation** (lines 246-370)

- `execute_plan()` function (lines 247-321):

- Initializes LangChain AgentExecutor with PythonREPLTool and DuckDuckGoSearchResults

- Executes plan steps sequentially, logging intermediate results

- Returns final output and all intermediate steps

- `generate_code_for_display_report()` function (lines 339-370):

- Takes agent's thoughtflow and generates display code (st.write, st.image)

- Code is editable in streamlit-ace editor in the "Code" tab

- Executed in the "AI Generated Report" panel to render final output

  

### Key Implementation Details

  

- **Session State Management** (lines 373-402): Custom `SessionStateAutoClass` provides auto-refresh on state changes. Use carefully as it triggers `st.rerun()` automatically.

  

- **Dynamic Code Execution**: User-editable report code is executed with `exec(reporting_code)` (line 512) to render the final report panel.

  

- **Agent Tools**:

- **PythonREPLTool** (lines 258-272): Executes Python code with Pandas, NumPy, Matplotlib. Saves plots as `plot.png`.

- **DuckDuckGoSearchResults** (lines 250-255): Web search for current information.

  

- **Data Flow**:

1. User asks chatbot → Intent detection (report creation vs simple question)

2. If report: Generate hierarchical plan with tool assignments

3. User clicks "Execute Plan" → AgentExecutor runs tools sequentially

4. AI generates display code from agent's thoughtflow

5. User can edit code in streamlit-ace → Execute to render report

  

### Technology Stack

  

**Main Branch**:

- **Frontend**: Streamlit 1.36.0 with layout="wide" for 4-panel grid

- **AI/Agent Framework**:

- LangChain 0.2.5 with AgentExecutor and OpenAI function calling

- langchain-experimental for PythonREPLTool and pandas agent

- LangSmith 0.1.81 for optional agent monitoring

- **Code Editing**: streamlit-ace 0.1.1 for in-browser Python editor

- **Chat UI**: streamlit-chat 0.1.1 for message bubbles

- **Data Processing**: Pandas 2.2.2, NumPy 1.26.4

- **Visualization**: Matplotlib 3.9.0

- **Search**: DuckDuckGo search integration

- **Environment**: python-dotenv for API key management

  

**V0.1 Branch** (additional):

- **Document Generation**: python-docx 1.1.2 for Word report export

- **PDF Processing**: pdfplumber 0.11.5 (unused in current implementation)

  

## Testing

  

**Main Branch**:

- The app initializes with a sample 3-column dataframe (columns A, B, C) for immediate testing

- Users can edit the dataframe directly in the bottom-left panel or upload CSV files

- Try prompts like: "Create a report calculating the correlation between columns B and C"

  

**V0.1 Branch**:

- Use sample CSV files in `demo_files_to_upload_manually/`:

- `mock_sensor_data2.csv`

- `mock_sensor_data3.csv`

- `motor_sensor_simulation_data.csv`

- These contain sensor data: Timestamp, Temperature, Vibration, Pressure, Flow Rate

  

## Important Notes

  

**Main Branch**:

- **Environment Setup**: `.env` file required with `OPENAI_API_KEY` (mandatory) and optional `LANGCHAIN_API_KEY` for monitoring

- **Hardcoded API Key Warning**: Line 185 and 215 contain a hardcoded OpenAI API key that should be replaced with `OPENAI_API_KEY` from environment

- **Plot Output**: Generated charts saved as `plot.png` in root directory

- **Agent Execution**: Uses real OpenAI API calls; costs may apply based on usage

- **LangSmith Monitoring**: Set `LANGCHAIN_TRACING_V2="true"` and `LANGCHAIN_PROJECT="data_analysis_copilot"` in code (lines 37-38) to enable

  

**V0.1 Branch**:

- AI responses are mocked with `time.sleep()` delays; no actual API calls made

- Charts saved as `chart.png`

- Generates Word (.docx) reports with embedded visualizations

  

**Both Branches**:

- `.env` file is gitignored and must be created locally

- `.github/` directory contains issue templates and PR template for collaboration