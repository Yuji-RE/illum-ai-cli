-- illum.lua — Neovim integration for illum explain
-- Usage: add to init.lua:
--   dofile("/path/to/illum-ai-cli/nvim/illum.lua")

local ILLUM = vim.fn.exepath("illum")
if ILLUM == "" then
  -- fallback to uv run in the project directory
  local project_dir = vim.fn.expand("~/projects/illum-ai-cli")
  ILLUM = string.format("cd %s && uv run illum", project_dir)
end

local function open_float(title)
  local width = math.floor(vim.o.columns * 0.7)
  local height = math.floor(vim.o.lines * 0.5)
  local row = math.floor((vim.o.lines - height) / 2)
  local col = math.floor((vim.o.columns - width) / 2)

  local buf = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_buf_set_option(buf, "bufhidden", "wipe")
  vim.api.nvim_buf_set_option(buf, "filetype", "markdown")

  local win = vim.api.nvim_open_win(buf, true, {
    relative = "editor",
    width = width,
    height = height,
    row = row,
    col = col,
    style = "minimal",
    border = "rounded",
    title = " " .. title .. " ",
    title_pos = "center",
  })

  vim.api.nvim_win_set_option(win, "wrap", true)
  vim.api.nvim_win_set_option(win, "linebreak", true)
  vim.keymap.set("n", "q", "<cmd>close<CR>", { buffer = buf, noremap = true, silent = true })

  return buf, win
end

local function run_explain(query)
  local buf, _ = open_float("illum explain: " .. query)
  local lines = { "" }

  local cmd = string.format("%s explain %s", ILLUM, vim.fn.shellescape(query))

  vim.fn.jobstart(cmd, {
    stdout_buffered = false,
    on_stdout = function(_, data)
      if not data then return end
      for _, chunk in ipairs(data) do
        if chunk ~= "" then
          -- append to last line, splitting on newlines
          local parts = vim.split(chunk, "\n", { plain = true })
          lines[#lines] = lines[#lines] .. parts[1]
          for i = 2, #parts do
            table.insert(lines, parts[i])
          end
          vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
        end
      end
    end,
    on_stderr = function(_, data)
      if not data then return end
      for _, line in ipairs(data) do
        if line ~= "" then
          table.insert(lines, "ERROR: " .. line)
        end
      end
      vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
    end,
  })
end

-- Normal mode: explain word under cursor
vim.keymap.set("n", "<leader>k", function()
  local word = vim.fn.expand("<cWORD>")
  run_explain(word)
end, { noremap = true, silent = true, desc = "illum explain (word)" })

-- Visual mode: explain selected text
vim.keymap.set("v", "<leader>k", function()
  -- yank selection into register v
  vim.cmd('noau normal! "vy')
  local sel = vim.fn.getreg("v")
  sel = sel:gsub("\n", " "):gsub("%s+", " "):gsub("^%s*(.-)%s*$", "%1")
  run_explain(sel)
end, { noremap = true, silent = true, desc = "illum explain (selection)" })
