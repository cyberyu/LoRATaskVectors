#!/usr/bin/env python3
"""
Script to diagnose the attached JSONL file for corruption.
"""

import json
import re

def analyze_jsonl_content():
    """Analyze the JSONL content from the attached file."""
    
    # The content from the attached file
    jsonl_lines = [
        '{"content":"<|fim_prefix|>#pragma once #include <strategy\\/stream\\/TCPStream.h> #include <random> #include \\"BinaryStreamBuffer.h\\" #include <simple-websocket-server\\/crypto.hpp> #include <simple-websocket-server\\/utility.hpp> class IWebsocketHandler { public: virtual ~IWebsocketHandler() = default; virtual void HandleWebsocketConnected() = 0; virtual void HandleWebsocketDisconnected(const tbricks::String & reason) = 0; virtual void HandleError(const tbricks::String & error) = 0; virtual void HandleMessage(std::string_view message) = 0; }; class TbWebsocketClient : public tbricks::TCPStream::IHandler { public: struct Config { \\/\\/\\/ Timeout on request handling. Defaults to no timeout. long timeout_request = 0; \\/\\/\\/ Idle timeout. Defaults to no timeout. long timeout_idle = 0; \\/\\/\\/ Maximum size of incoming messages. Defaults to architecture maximum. \\/\\/\\/ Exceeding this limit will result in a message_size error code and the connection will be closed. std::size_t max_message_size = (std::numeric_limits<std::size_t>::max)(); \\/\\/\\/ Additional header fields to send when performing WebSocket upgrade. \\/\\/\\/ Use this variable to for instance set Sec-WebSocket-Protocol. SimpleWeb::CaseInsensitiveMultimap header; \\/\\/\\/ Set proxy server (server:port) std::string proxy_server; \\/\\/\\/ Set proxy authorization (username:password) std::string proxy_auth; unsigned short default_port = 80; }; public: TbWebsocketClient(IWebsocketHandler & handler, const tbricks::String & host_port_path, const TbWebsocketClient::Config & conf); ~TbWebsocketClient(); void Connect(); void Shutdown(int status = 1000, const std::string & reason = \\"\\"); \\/\\/\\/ fin_rsv_opcode: 129=one fragment, text, 130=one fragment, binary, 136=close<|fim_suffix|><|fim_middle|>connection. \\/\\/\\/ See http:\\/\\/tools.ietf.org\\/html\\/rfc6455#section-5.2 for more information.     void SendWs(const tbricks::Binary & buffer, unsigned char fin_rsv_opcode = 129);"}',
        '{"content":"<|fim_prefix|>#include \\"InstrumentManager.h\\"\\n\\nusing namespace tbricks;\\n\\nInstrumentManager::InstrumentManager(IHandler& handler)\\n    : m_handler(handler)\\n    , m_instrumentStream(*this)\\n    , m_iapStream(*this)\\n    , m_calculatedValuesStream(*this)\\n    , m_failed(0)\\n<|fim_suffix|><|fim_middle|>{\\n    TBDEBUG(__func__ << \\" ctor\\");\\n}"}',
        '{"content":"<|fim_prefix|>#include <optional> #include \\"DeribitMdp.h\\" #include \\"tbricks_definitions.h\\" using namespace tbricks; DeribitMdp::DeribitMdp() { TBDEBUG(\\"Constructor: \\" << GetVenueIdentifier()); } void DeribitMdp::HandleSubscribe(MarketDataItem & item) { TBDEBUG(__func__ << \\", item = \\" << item); const auto & instrument_id = item.GetInstrumentVenueIdentification().GetInstrumentIdentifier(); auto model = DeribitModel(item, *this); m_models.emplace(instrument_id, model); m_streams.emplace(model.GetStreamId(), instrument_id); } void DeribitMdp::HandleUnsubscribe(MarketDataItem & item) { TBDEBUG(__func__ << \\", item = \\" << item); const auto & instrument_id = item.GetInstrumentVenueIdentification().GetInstrumentIdentifier(); auto model_opt = GetModel(instrument_id); if (model_opt) { auto * model = *model_opt; m_streams.erase(model->GetStreamId()); m_models.erase(instrument_id); } } void DeribitMdp::HandleOrderBookSubscribe(OrderBookItem & item) { TBDEBUG(__func__ << \\", item = \\" << item); } void DeribitMdp::HandleOrderBookUnsubscribe(OrderBookItem & item) { TBDEBUG(__func__ << \\", item = \\" << item); } bool DeribitMdp::IsMarketDataSupported(const InstrumentVenueIdentification &<|fim_suffix|>{ TBDEBUG(__func__ << \\", ivid = \\" << ivid); return true; } void DeribitMdp::GetDiagnostics(Diagnostics & diagnostics) { diagnostics.GetDebugStream() << \\"Market aggregator diagnostic \\"; if (!diagnostics.GetDiagnosticKey().empty()) { { diagnostics.GetDebugStream() << \\"for instrument \\" << diagnostics.GetDiagnosticKey(); } } diagnostics.GetDebugStream() << std::endl; } void DeribitMdp::HandleStreamOpen(const StreamIdentifier & stream) { TBDEBUG(__func__ << \\": \\" << stream); } void DeribitMdp::HandleStreamStale(const StreamIdentifier & stream) { TBDEBUG(__func__ << \\": \\" << stream); } void DeribitMdp::HandleStreamFailed(const StreamIdentifier & stream) { TBDEBUG(__func__ << \\": \\" << stream); } void DeribitMdp::HandleSnapshotEnd(const StreamIdentifier & stream)<|fim_middle|>ivid)"}',
        '{"content":"<|fim_prefix|>VOLATILITY_MODEL_WRAP(SVI,StochasticVolatilityInspired)<|fim_suffix|><|fim_middle|>VOLATILITY_MODEL_WRAP(CCS,ClampedCubicSpline) VOLATILITY_MODEL_WRAP(Wing,WingVolatilityModel)"}',
        '{"content":"<|fim_prefix|>#include <shared\\/ValueWithDefault.hpp> #include <shared\\/SuppressSfApiClangWarnings.h> SUPPRESS_SF_API_CLANG_WARNINGS #include <strategy\\/type\\/Price.h> #include <strategy\\/type\\/Integer.h> #include <strategy\\/type\\/Boolean.h> #include <strategy\\/calculated_property\\/CalculatedPropertyFilter.h> #include <strategy\\/TreeNodeParameters.h> #include <strategy\\/parameter\\/StrategyParameters.h> #include <strategy\\/parameter\\/StrategyInstanceParameters.h> CLANG_RESTORE_WARNINGS #include <set> using namespace tbricks; enum class PreferencesStorageKind { Common = 0, Wing, CCS, SVI, MyCustom = 100000 }; std::ostream & operator<<(std::ostream & strm, const PreferencesStorageKind kind); \\/** * Struct representing the various settings common to all VMs; it is subclassed by structs * containing additional parameters relevant for particular VMs *\\/ class PreferencesStorage: public Printable { public: PreferencesStorage(); protected: PreferencesStorage(const PreferencesStorage & storage) = default; PreferencesStorage & operator=(const PreferencesStorage & storage) = default; public: virtual ~PreferencesStorage(); virtual PreferencesStorageKind GetKind() const = 0; virtual void Clear(); virtual void Copy(const PreferencesStorage & preferencesStorage); virtual bool FillFromTreeNodeParameters(const TreeNodeParameters & parameters); virtual bool FillFromStrategyParameters(const StrategyParameters & parameters); virtual bool FillFromStrategyInstanceParameters(const StrategyInstanceParameters & parameters); virtual bool GetParameter(const TreeNodeParameterDefinition & def, AnyType & value) const; virtual bool GetParameter(const ParameterDefinition & def, AnyType & value) const; virtual void GetParameters(TreeNodeParameters & parameters) const; virtual void GetParameters(StrategyParameters & parameters) const; virtual void GetParameters(StrategyInstanceParameters & parameters) const; virtual bool GetNonDefaultParameter(const TreeNodeParameterDefinition & def, AnyType & value) const; virtual bool GetNonDefaultParameter(const ParameterDefinition & def, AnyType &<|fim_suffix|>virtual void GetNonDefaultParameters(StrategyParameters & parameters) const; virtual void GetNonDefaultParameters(StrategyInstanceParameters & parameters) const; \\/\\/ Update local values with non-empty values from the passed in preferences storage; \\/\\/ return true if something got changed as a result, false otherwise virtual bool Merge(const PreferencesStorage & preferencesStorage); virtual std::ostream & Print(std::ostream & strm) const override; \\/\\/ List of preferences that shouldn\\'t be copied to other scope levels virtual const std::set<TreeNodeParameterDefinition> & GetLevelAnchoredParameters() const { return m_scopeLevelAnchoredPreferences; } \\/\\/ In case<|fim_middle|>value) const; virtual void GetNonDefaultParameters(TreeNodeParameters & parameters) const;"}',
        '{"content":"<|fim_prefix|>#include \\"BinaryStreamBuffer.h\\"\\n\\ntemplate <>\\nBinary_istream & operator>>(Binary_istream & istm, std::string & val)\\n{\\n    int size = 0;\\n    istm.read(size);\\n\\n    if (size <= 0)\\n        return istm;\\n\\n    istm.read(val, size);\\n\\n    return istm;\\n}\\n\\ntemplate <>\\nvoid Binary_istream::read(tbricks::Binary & out)\\n{\\n    out = m_data;\\n}\\n\\ntemplate <>\\nvoid Binary_ostream::write(const tbricks::Binary & in)\\n{\\n    m_data.Append(in);\\n}\\n\\ntemplate <>\\nBinary_ostream & operator<<(Binary_ostream & ostm, const std::string & val)\\n{\\n    if (val.size() <= 0)\\n        return ostm;\\n\\n    ostm.write(val.c_str(), val.size());\\n\\n    return ostm;\\n}\\n\\nBinary_ostream & operator<<(Binary_ostream & ostm, const char * val)\\n{\\n    int size = std::strlen(val);\\n    if (size <= 0)\\n        return ostm;\\n\\n    ostm.write(val, size);\\n\\n    return ostm;\\n}\\n\\nBinary_ostream & operator<<(Binary_ostream & ostm, const char val)\\n<|fim_suffix|><|fim_middle|>{\\n    ostm.write(&val, sizeof(char));"}',
        '{"content":"<|fim_prefix|>#pragma once #include <strategy\\/stream\\/CalculatedPropertiesStream.h> #include <vector> class PositionsData<|fim_suffix|>{ public: PositionsData(const tbricks::PortfolioIdentifier& portfolioId, tbricks::StringParameter& status); ~PositionsData(); void Stop(); public: using Data =<|fim_middle|>: public tbricks::CalculatedPropertiesTable::Stream::IHandler"}',
        '{"content":"<|fim_prefix|>#include \\"InstrumentEnricherPlugin.h\\" #include \\"tbricks_definitions.h\\" using namespace tbricks; InstrumentEnricherPlugin::InstrumentEnricherPlugin( const InitializationReason & reason, const StrategyParameters & parameters) : m_instrumentGroup(instrument_enricher::strategy_parameters::InstrumentGroup()) , m_maturityDate(instrument_enricher::strategy_parameters::MaturityDate()) , m_plugInStatusDescription(instrument_enricher::strategy_parameters::PlugInStatusDescription()) , m_underlyingInstrument(instrument_enricher::strategy_parameters::UnderlyingInstrument()) { TBSTATUS(\\"Constructor: \\" << reason); TBDEBUG(\\"Parameters: \\" << parameters); \\/\\/ Accept all parameter suggestions GetParameters().Merge(parameters); \\/\\/ Always start in paused mode SetState(StrategyState::PAUSED); } void InstrumentEnricherPlugin::HandleRunRequest() { TBDEBUG(\\"HandleRunRequest\\"); SetState(StrategyState::RUNNING); } void InstrumentEnricherPlugin::HandlePauseRequest() { TBDEBUG(\\"HandlePauseRequest\\"); SetState(StrategyState::PAUSED); } void InstrumentEnricherPlugin::HandleDeleteRequest() { TBDEBUG(\\"HandleDeleteRequest\\"); SetState(StrategyState::DELETED); } void InstrumentEnricherPlugin::HandleModifyRequest(const StrategyModifier & modifier) { TBSTATUS(\\"Modify request: \\" << modifier); \\/\\/ Accept all modified parameters and attributes GetParameters().Merge(modifier.GetParameters()); MergeAttributes(modifier.GetAttributes()); } void InstrumentEnricherPlugin::HandleValidateRequest(ValidationContext &context) { TBDEBUG(\\"HandleValidateRequest: \\" << context); \\/\\/ No default validation so we just inform the frontend that<|fim_suffix|><|fim_middle|>validation is completed     context.SendReply();"}',
        '{"content":"<|fim_prefix|>#pragma once #include \\"OrderExecutor.h\\" #include \\"shared\\/order_minion\\/OrderMinionController.h\\" #include \\"shared\\/order_minion\\/OrderMinionRequest.h\\" namespace execution { class TbOrderMinion : public OrderExecutor,<|fim_suffix|><|fim_middle|>public OrderMinionRequest::IHandler"}'
    ]
    
    print("🔍 JSONL File Analysis")
    print("=" * 50)
    
    valid_lines = 0
    corrupted_lines = 0
    issues = []
    
    for i, line in enumerate(jsonl_lines, 1):
        try:
            # Try to parse as JSON
            json_obj = json.loads(line)
            valid_lines += 1
            
            # Check content structure
            if not isinstance(json_obj, dict):
                issues.append(f"Line {i}: Not a JSON object")
            elif 'content' not in json_obj:
                issues.append(f"Line {i}: Missing 'content' field")
            else:
                content = json_obj['content']
                
                # Check for FIM tokens
                fim_tokens = ['<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>']
                token_counts = {token: content.count(token) for token in fim_tokens}
                
                # Validate FIM structure
                if token_counts['<|fim_prefix|>'] != 1:
                    issues.append(f"Line {i}: Expected 1 fim_prefix, found {token_counts['<|fim_prefix|>']}")
                if token_counts['<|fim_suffix|>'] != 1:
                    issues.append(f"Line {i}: Expected 1 fim_suffix, found {token_counts['<|fim_suffix|>']}")
                if token_counts['<|fim_middle|>'] != 1:
                    issues.append(f"Line {i}: Expected 1 fim_middle, found {token_counts['<|fim_middle|>']}")
                
                # Check for escaped characters that might cause issues
                if '\\\\' in content:
                    backslash_count = content.count('\\\\')
                    if backslash_count > 50:  # Arbitrary threshold
                        issues.append(f"Line {i}: High number of escaped backslashes ({backslash_count})")
                
                # Check for unescaped quotes
                unescaped_quotes = len(re.findall(r'(?<!\\)"', content))
                if unescaped_quotes > 2:  # More than opening and closing quotes
                    issues.append(f"Line {i}: Potential unescaped quotes detected")
                
        except json.JSONDecodeError as e:
            corrupted_lines += 1
            issues.append(f"Line {i}: JSON decode error - {str(e)}")
        except Exception as e:
            corrupted_lines += 1
            issues.append(f"Line {i}: Unexpected error - {str(e)}")
    
    print(f"📊 Results:")
    print(f"✅ Valid lines: {valid_lines}")
    print(f"❌ Corrupted lines: {corrupted_lines}")
    print(f"⚠️  Issues found: {len(issues)}")
    
    if issues:
        print(f"\n🔍 Detailed Issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  • {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    
    # Check specific patterns that might indicate corruption
    print(f"\n🔧 Corruption Analysis:")
    
    # Look for common corruption patterns
    corruption_patterns = []
    
    for i, line in enumerate(jsonl_lines, 1):
        # Check for truncated lines (lines that end abruptly)
        if not line.strip().endswith('}'):
            corruption_patterns.append(f"Line {i}: Doesn't end with '}}' - possibly truncated")
        
        # Check for extremely long lines (might indicate concatenation issues)
        if len(line) > 10000:
            corruption_patterns.append(f"Line {i}: Extremely long line ({len(line)} chars)")
        
        # Check for null bytes or control characters
        if '\x00' in line:
            corruption_patterns.append(f"Line {i}: Contains null bytes")
    
    if corruption_patterns:
        print("  Potential corruption patterns:")
        for pattern in corruption_patterns:
            print(f"    • {pattern}")
    else:
        print("  No common corruption patterns detected")
    
    return valid_lines, corrupted_lines, issues

if __name__ == "__main__":
    analyze_jsonl_content()
