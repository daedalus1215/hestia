import {
  Client,
  ToolResult,
  types,
} from "@modelcontextprotocol/sdk/client/index.js";
import { resolve } from "path";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

// Configuration
const SERVER_PATH = resolve(
  "/home/dev/programming/hestia/mcp-servers/quickstart-resources/weather-server-typescript/build/index.js"
); // Update this path
const LLAMA3_API_URL = "http://localhost:11434/api/generate"; // Adjust if your Llama3 API is different

// Initialize MCP client
const client = new Client({
  capabilities: {
    prompt: true,
    toolCall: true,
    toolResult: true,
  },
});

// Function to query Llama3 via Ollama API (or adapt for your Llama3 setup)
async function queryLlama3(
  prompt: string,
  tools: types.Tool[]
): Promise<string> {
  const response = await fetch(LLAMA3_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "llama3", // Adjust to your Llama3 model name
      prompt: `${prompt}\n\nAvailable tools:\n${JSON.stringify(
        tools,
        null,
        2
      )}`,
      stream: false,
    }),
  });

  const data = await response.json();
  return data.response || "No response from Llama3";
}

// Main function to run the client
async function main() {
  // Start the MCP server process
  const serverProcess = exec(`node ${SERVER_PATH}`);

  // Connect to the MCP server
  await client.connect({
    command: "node",
    args: [SERVER_PATH],
  });

  // List available tools
  const tools = await client.listTools();
  console.log("Available tools:", tools);

  // Query Llama3 with a prompt to get SF weather
  const prompt = `Please get the weather forecast for San Francisco, CA (latitude: 37.7749, longitude: -122.4194) using the available tools.`;
  const response = await queryLlama3(prompt, tools);

  console.log("Llama3 Response:", response);

  // Execute the tool call (assuming Llama3 suggests a tool call)
  // This is a simplified example; Llama3 may not natively parse tools like Claude
  const toolCall: types.ToolCall = {
    id: "tool_call_1",
    name: "get-forecast",
    arguments: {
      latitude: 37.7749,
      longitude: -122.4194,
    },
  };

  const toolResult: ToolResult = await client.callTool(toolCall);
  console.log("Tool Result:", JSON.stringify(toolResult, null, 2));

  // Clean up
  serverProcess.kill();
  await client.disconnect();
}

main().catch(console.error);
