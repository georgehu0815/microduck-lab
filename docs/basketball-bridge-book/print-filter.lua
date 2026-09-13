function Image(image)
  if FORMAT:match("latex") then
    image.attributes.width = "6.2in"
    image.attributes.height = "5.5in"
  end
  return image
end

function Header(header)
  if FORMAT:match("latex") and #header.content > 1 and header.content[1].t == "Str"
      and header.content[1].text:match("^%d[%d%.]*%.?$") then
    table.remove(header.content, 1)
    if header.content[1].t == "Space" then
      table.remove(header.content, 1)
    end
  end
  return header
end

local function has_cjk(text)
  for _, point in utf8.codes(text) do
    if (point >= 0x2E80 and point <= 0x9FFF) or (point >= 0xFF00 and point <= 0xFFEF) then
      return true
    end
  end
  return false
end

function CodeBlock(block)
  if FORMAT:match("latex") and has_cjk(block.text) then
    return {pandoc.RawBlock("latex", "\\begingroup\\setmonofont{PingFang SC}"),
            block, pandoc.RawBlock("latex", "\\endgroup")}
  end
  return block
end

function Code(code)
  if FORMAT:match("latex") and has_cjk(code.text) then
    return pandoc.Span({pandoc.RawInline("latex", "{\\setmonofont{PingFang SC}"),
                        code, pandoc.RawInline("latex", "}")})
  end
  if FORMAT:match("latex") and #code.text > 12 and code.text:match("^[%w%._/%-]+$") then
    local pieces = {}
    local count = 0
    for character in code.text:gmatch(".") do
      table.insert(pieces, character == "_" and "\\_" or character)
      count = count + 1
      if character:match("[_/%-]") or count == 8 then
        table.insert(pieces, "\\allowbreak{}")
        count = 0
      end
    end
    return pandoc.RawInline("latex", "\\texttt{" .. table.concat(pieces) .. "}")
  end
  return code
end
