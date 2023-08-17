import streamlit as st
# Adopted from Streamlit Hello multipage guide
# Source : https://github.com/streamlit/hello/blob/main/Hello.py

def run():
    st.set_page_config(
        page_title="Computer Vision",
        page_icon="👋",
    )

    st.write("## Computer Vision Projects Homepage! 👋")

    st.sidebar.success("Select a project above for live demo.")

    st.markdown(
        """
        This page is built specifically to showcase my 
        Computer Vision projects.
        .
        
        **👈 Select a demo from the sidebar** to see some examples
        of the projects I have worked on!
        ### Want to get in touch?
        - Add me on [Linkedin](https://www.linkedin.com/in/mchockal/)
        - Find me on [Github](https://github.com/mchockal)
        - Bookmark [my blog](https://mchockal.github.io/) to keep up with my updates 
    """
    )


if __name__ == "__main__":
    run()