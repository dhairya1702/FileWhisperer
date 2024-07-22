document.getElementById('chatForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const project_name = document.getElementById('project_name').value;
    const question = document.getElementById('question').value;

  const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
            'project_name': project_name,
            'question': question
        })
    });

    const result = await response.json();


    // Simulate with a dummy answer
    const dummyResponse = {
        answer: `This is a dummy response for project: ${project_name}, question: ${question}`
    };


    // Change dummyResponse to response to get GPT given answer
    document.getElementById('response').innerText = result.answer;
});
