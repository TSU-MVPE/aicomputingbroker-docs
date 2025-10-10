// 初始化 Mermaid
document.addEventListener('DOMContentLoaded', function() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            theme: 'default',
            themeVariables: {
                primaryColor: '#1976d2',
                primaryTextColor: '#ffffff'
            },
            securityLevel: 'loose',
            startOnLoad: false,  // Set to false, manually control rendering
            htmlLabels: true,
            flowchart: {
                htmlLabels: true
            }
        });

        // Find all pre elements containing mermaid
        const mermaidElements = document.querySelectorAll('pre.mermaid');
        mermaidElements.forEach((element, index) => {
            // Extract content from pre > code
            const codeElement = element.querySelector('code');
            const graphDefinition = codeElement ? codeElement.textContent.trim() : element.textContent.trim();
            const graphId = 'mermaid-' + index;
            
            // Create a new div to replace the pre element
            const mermaidDiv = document.createElement('div');
            mermaidDiv.className = 'mermaid';
            mermaidDiv.id = graphId;
            
            try {
                // Use the new mermaid.render API
                mermaid.render(graphId + '-svg', graphDefinition).then(function(result) {
                    mermaidDiv.innerHTML = result.svg;
                    element.parentNode.replaceChild(mermaidDiv, element);
                }).catch(function(error) {
                    console.error('Mermaid rendering error:', error);
                    mermaidDiv.innerHTML = '<p style="color: red;">Mermaid rendering failed: ' + error.message + '</p>';
                    element.parentNode.replaceChild(mermaidDiv, element);
                });
            } catch (error) {
                console.error('Mermaid rendering error:', error);
                mermaidDiv.innerHTML = '<p style="color: red;">Mermaid rendering failed: ' + error.message + '</p>';
                element.parentNode.replaceChild(mermaidDiv, element);
            }
        });
    } else {
        console.error('Mermaid library not loaded');
    }
});
