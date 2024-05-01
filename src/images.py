from playwright.sync_api import sync_playwright

style = """<style>
            * {
                font-family: 'IBM Plex Sans' !important;
                font-weight: normal !important;
                font-style: normal !important;
            }

            h1 {
               font-size: calc(1.375rem + 1.5vw * 1.75)!important
            }
            .fs-3 {
                font-size: calc(1.3rem + .6vw * 1.75)!important;
            }

            .fs-4 {
                font-size: calc(1.275rem + .3vw * 1.75)!important;
            }

            .fs-6, span {
                font-size: calc(1rem * 1.75) !important;
            }
        </style>"""


def generateImage(title,reddit, outDir):
    html_content = f"""

<!DOCTYPE html>
        <html>
        <head>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&display=swap" rel="stylesheet">
        
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
        {style}
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

        <title>HTML to Image</title>
        </head>
        <body>
            <div style="min-height: 100vh; min-width: 100vw;" class="position-relative bg-black">
                <div class="position-absolute ">
                    <img src="https://storage.googleapis.com/linebox-bucket/pexels-merlin-lightpainting-10874585.jpg" style="height: 100vh; width: 100vw; position: absolute;; filter: blur(2px)"/>
                </div>
                <div style="min-height: 100vh;" class="d-flex justify-content-center align-items-center position-relative">
                    <div class="container card m-5 p-3">
                        <div class="d-flex gap-4">
                            <img height="100" class="rounded-5" width="100" src="https://styles.redditmedia.com/t5_2r0cn/styles/communityIcon_qadm8xvply981.png"/>
                           
                           <div>

                               <h1 class="fs-3">r/{reddit}
                               </h1>
                               <h1 class="fs-4 fw-light">@gh0stpalace
                                <i class="bi bi-check-circle text-primary"></i>
   
                               </h1>
                           </div>
                        </div>
                         <h1 class="fw-bold my-5">{title}</h1>
                        <div class="d-flex gap-4">

                            <div  style="background-color: #eaedef;" class=" rounded-5 d-flex align-items-center gap-4 px-4 py-2">
                                <span class="">
                                <i class="bi bi-heart fs-6"></i>
                                    99</span>
                            </div>

                            <div style="background-color: #eaedef;" class=" rounded-5 d-flex align-items-center gap-4 px-4 py-2">

                                <span class="" >
                                    <i class="bi bi-card-text fs-6"></i>
                                        99</span>

                            </div>

                        </div>
                    </div>
                </div>
            </div>
        </body>
    </html>
    
    
    """

    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch()

        page = browser.new_page()

        page.set_content(html_content)

        page.wait_for_load_state()
        page.screenshot(path=outDir+"/frontpage.png", type="png")
        page.set_viewport_size({"width": 1920, "height": 1080})  # Adjust based on your desired size

        browser.close()
        return
       

#generateImage("I (F24) don’t know if I should be concerned about these weird videos I found in my bf’s (M33) camera roll?", "relationship_advice", "outnwe")
